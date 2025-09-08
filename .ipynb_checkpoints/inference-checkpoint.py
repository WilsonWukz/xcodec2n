"""
Robust inference script with auto-detected checkpoint dims and channel mapping:
 - auto-detects decoder config from checkpoint (vq_dim, fc dims)
 - adds 1x1 conv if encoder/decoder channel mismatch
 - safe decoding fallback for layout issues
Usage example:
python inference.py \
  --input-dir /workspace/xcodec2n/test_audio/input_test \
  --ckpt '/workspace/xcodec2n/outputs/checkpoints/seed1024.ckpt' \
  --output-dir /workspace/xcodec2n/test_audio/output_test \
  --device cuda \
  --sr 16000
"""
import os
import argparse
import torch
import torch.nn as nn
from glob import glob
from tqdm import tqdm
import warnings

# repo modules
from vq.codec_encoder import CodecEncoder
try:
    from vq.codec_decoder_vocos_o import CodecDecoderVocos as CodecDecoderVocos_o
except Exception:
    CodecDecoderVocos_o = None
try:
    from vq.codec_decoder_vocos import CodecDecoderVocos as CodecDecoderVocos_old
except Exception:
    CodecDecoderVocos_old = None
from vq.module import SemanticEncoder
from transformers import Wav2Vec2BertModel

DEFAULT_VQ_DIM = 256

def filter_state_dict(state_dict):
    codec, gen, fc_post_a, sem_enc, fc_prior = {}, {}, {}, {}, {}
    for k, v in state_dict.items():
        if k.startswith('CodecEnc.'):
            codec[k[len('CodecEnc.'):]] = v
        elif k.startswith('generator.'):
            gen[k[len('generator.'):]] = v
        elif k.startswith('fc_post_a.'):
            fc_post_a[k[len('fc_post_a.'):]] = v
        elif k.startswith('SemanticEncoder_module.'):
            sem_enc[k[len('SemanticEncoder_module.'):]] = v
        elif k.startswith('fc_prior.'):
            fc_prior[k[len('fc_prior.'):]] = v
    return codec, gen, fc_post_a, sem_enc, fc_prior

def try_load_ckpt(ckpt_path, map_location='cpu'):
    try:
        return torch.load(ckpt_path, map_location=map_location, weights_only=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

def infer_decoder_and_fc_dims(gen_sd, fc_post_a_sd, fc_prior_sd):
    info = {}
    # vq_dim
    if 'quantizer.project_out.weight' in gen_sd:
        info['vq_dim'] = gen_sd['quantizer.project_out.weight'].shape[0]
    elif 'quantizer.project_in.weight' in gen_sd:
        info['vq_dim'] = gen_sd['quantizer.project_in.weight'].shape[1]
    else:
        info['vq_dim'] = DEFAULT_VQ_DIM
    # fc_post_a / fc_prior shapes
    for sd, name in [(fc_post_a_sd, 'fc_post_a_weight_shape'), (fc_prior_sd, 'fc_prior_weight_shape')]:
        if len(sd) > 0:
            for k, v in sd.items():
                if k.endswith('weight'):
                    info[name] = tuple(v.shape)
                    break
    return info

def instantiate_modules_auto(device, gen_info):
    encoder = CodecEncoder().eval().to(device)

    # decoder class
    dec_cls = CodecDecoderVocos_o or CodecDecoderVocos_old
    if dec_cls is None:
        raise RuntimeError("No decoder class found")

    # decoder kwargs
    kwargs = {}
    if 'vq_dim' in gen_info:
        kwargs['vq_dim'] = int(gen_info['vq_dim'])
    kwargs['hop_length'] = 320
    decoder = dec_cls(**kwargs).eval().to(device)

    # semantic modules
    semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0",
                                                       output_hidden_states=True).eval().to(device)
    sem_enc = SemanticEncoder(1024, 1024, 1024).eval().to(device)

    # fc layers
    if 'fc_prior_weight_shape' in gen_info:
        out, inp = gen_info['fc_prior_weight_shape']
        fc_prior = nn.Linear(inp, out).eval().to(device)
    else:
        fc_prior = nn.Linear(1024, 2048).eval().to(device)
    if 'fc_post_a_weight_shape' in gen_info:
        out, inp = gen_info['fc_post_a_weight_shape']
        fc_post_a = nn.Linear(inp, out).eval().to(device)
    else:
        fc_post_a = nn.Linear(2048, 1024).eval().to(device)

    return encoder, decoder, semantic_model, sem_enc, fc_prior, fc_post_a

def load_filtered_state_dicts(modules, state_dict):
    encoder, decoder, semantic_model, sem_enc, fc_prior, fc_post_a = modules
    codec_sd, gen_sd, fc_post_a_sd, sem_enc_sd, fc_prior_sd = filter_state_dict(state_dict)

    try:
        encoder.load_state_dict(codec_sd, strict=False)
        decoder.load_state_dict(gen_sd, strict=False)
        sem_enc.load_state_dict(sem_enc_sd, strict=False)
        if fc_post_a_sd:
            out, inp = fc_post_a_sd['weight'].shape
            fc_post_a = nn.Linear(inp, out).to(next(encoder.parameters()).device)
            fc_post_a.load_state_dict(fc_post_a_sd, strict=False)
        if fc_prior_sd:
            out, inp = fc_prior_sd['weight'].shape
            fc_prior = nn.Linear(inp, out).to(next(encoder.parameters()).device)
            fc_prior.load_state_dict(fc_prior_sd, strict=False)
    except Exception as e:
        warnings.warn(f"Loading weights encountered: {e}")
    return encoder, decoder, semantic_model, sem_enc, fc_prior, fc_post_a

def safe_decode(decoder, prior_emb, layout='bct'):
    if layout == 'btc' and prior_emb.shape[1] > prior_emb.shape[2]:
        prior_emb = prior_emb.transpose(1, 2).contiguous()
    try:
        return decoder(prior_emb, vq=False)
    except:
        try:
            return decoder(prior_emb.transpose(1, 2).contiguous(), vq=False)
        except Exception as e:
            raise RuntimeError(f"Decoder failed: {e}")

def build_models_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt['state_dict']  # 或者直接是 ckpt，看你保存的结构
    print("All keys in checkpoint:")
    for k in state_dict.keys():
        print(k)
    
    # 取 encoder 输出通道数
    enc_out_channels = state_dict['CodecEnc.conv_blocks.0.weight_v'].shape[0]
    
    # 取 decoder第一层输入通道数
    dec_in_channels = state_dict['CodecDec.conv_blocks.0.weight_v'].shape[1]
    
    print(f"Encoder out channels: {enc_out_channels}, Decoder in channels: {dec_in_channels}")

    codec_sd, gen_sd, fc_post_a_sd, sem_enc_sd, fc_prior_sd = filter_state_dict(state_dict)
    gen_info = infer_decoder_and_fc_dims(gen_sd, fc_post_a_sd, fc_prior_sd)
    encoder, decoder, semantic_model, sem_enc, fc_prior, fc_post_a = instantiate_modules_auto(device, gen_info)
    
    # 自动添加1x1 conv映射通道
    channel_map = None
    if enc_out_channels != dec_in_channels:
        print(f"Adding 1x1 conv: {enc_out_channels} -> {dec_in_channels}")
        channel_map = nn.Conv1d(enc_out_channels, dec_in_channels, kernel_size=1).to(device)
    
    encoder, decoder, semantic_model, sem_enc, fc_prior, fc_post_a = load_filtered_state_dicts(
        (encoder, decoder, semantic_model, sem_enc, fc_prior, fc_post_a),
        state_dict
    )
    detected_layout = 'bct'
    return encoder, decoder, sem_enc, fc_post_a, fc_prior, detected_layout, channel_map

def run_single_inference(encoder, decoder, sem_enc, fc_post_a, fc_prior,
                         input_path, output_path, device, sr, layout, channel_map=None):
    import torchaudio
    wav, in_sr = torchaudio.load(input_path)
    if in_sr != sr:
        wav = torchaudio.functional.resample(wav, in_sr, sr)
    wav = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        enc_out = encoder(wav)
        # 通道映射
        if channel_map is not None:
            enc_out = channel_map(enc_out)
    if enc_out.dim() == 3 and enc_out.shape[1] < enc_out.shape[2]:
        enc_out = enc_out.transpose(1, 2).contiguous()
    prior_emb = fc_prior(enc_out.transpose(1, 2)).transpose(1, 2) if fc_prior else enc_out
    audio = safe_decode(decoder, prior_emb, layout)
    torchaudio.save(output_path, audio.cpu(), sr)
    print(f"[ok] wrote {output_path}")

def run_inference(args):
    state_dict = try_load_ckpt(args.ckpt, map_location='cpu')
    device = torch.device(args.device)
    encoder, decoder, sem_enc, fc_post_a, fc_prior, layout, channel_map = build_models_from_ckpt(args.ckpt, device)
    input_files = glob(os.path.join(args.input_dir, "*.wav")) + glob(os.path.join(args.input_dir, "*.flac"))
    os.makedirs(args.output_dir, exist_ok=True)
    for path in tqdm(input_files):
        fname = os.path.basename(path)
        out_path = os.path.join(args.output_dir, f"{getattr(args, 'ckpt_prefix','')}{fname}")
        run_single_inference(encoder, decoder, sem_enc, fc_post_a, fc_prior, path, out_path, device, args.sr, layout, channel_map)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default='test_audio/output_test')
    parser.add_argument('--sr', type=int, default=16000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--hop', type=int, default=320)
    parser.add_argument('--ckpt_prefix', type=str, default="")
    args = parser.parse_args()
    run_inference(args)
