import soundfile as sf
in_wav = '/workspace/xcodec2n/test_audio/input_test/test.flac'
out_wav = '/workspace/xcodec2n/test_audio/output_test/seed=1024_n_10_3_4_f_20_10___test.flac'
yin, sr1 = sf.read(in_wav)
yout, sr2 = sf.read(out_wav)
print("sr_in, sr_out:", sr1, sr2)
print("len_in, len_out (samples):", len(yin), len(yout))
print("RMS_in, RMS_out:", (yin**2).mean()**0.5, (yout**2).mean()**0.5)

import torch

ckpt = torch.load(
    "/workspace/xcodec2n/outputs/checkpoints/seed=1024{n, 10, 3, 4, f, 20, 10}.ckpt",
    map_location='cpu',
    weights_only=False
)

for k, v in ckpt.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)
    else:
        print(k, type(v))
import torch

# 替换成你的 ckpt 路径
ckpt_path = '/workspace/xcodec2n/outputs/checkpoints/seed=1024{n, 10, 3, 4, f, 20, 10}.ckpt'

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

state_dict = ckpt['state_dict']  # 取出权重

# 找 decoder 相关权重 key，通常包含 'dec' 或 'CodecDec'
dec_keys = [k for k in state_dict.keys() if "dec" in k.lower() and "weight" in k.lower()]
print("找到的 decoder 权重 keys：")
for k in dec_keys:
    print(k)

# 假设第一层就是你要的
first_dec_key = dec_keys[0]
tensor = state_dict[first_dec_key]
print(f"\n第一层 decoder 权重 key: {first_dec_key}")
print("权重 shape:", tensor.shape)

# 输入通道数
in_channels = tensor.shape[1]
print("第一层 decoder 输入通道数:", in_channels)




