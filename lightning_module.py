import os
 
import random
import hydra
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from vq import CodecEncoder,  CodecDecoderVocos 
from module import HiFiGANMultiPeriodDiscriminator, SpecDiscriminator
from criterions import GANLoss, MultiResolutionMelSpectrogramLoss, MultiResolutionSTFTLoss
from common.schedulers import WarmupLR
from transformers import AutoModel
from vq.module import SemanticDecoder,SemanticEncoder
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel
import sys
sys.path.append('./eval_tools/tools/speaker_verification')    # We use wavlm_large_finetune as a vadidation metric during training, https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification
from  verification import init_model
model_spk = init_model('wavlm_large','/workspace/xcodec2n/wavlm_large.pt')



class CodecLightningModule(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # self.ocwd = hydra.utils.get_original_cwd()
        try:  
            self.ocwd = hydra.utils.get_original_cwd()  
        except ValueError:  
            self.ocwd = os.getcwd() 
        self.construct_model()
        self.construct_criteria()
        self.save_hyperparameters()
        self.automatic_optimization = False

    def construct_model(self):
        # 初始化 Codec Encoder
 
        enccfg = self.cfg.model.codec_encoder

 
        self.CodecEnc = CodecEncoder(
 
            ngf=enccfg.ngf,
            up_ratios=enccfg.up_ratios,
            dilations=enccfg.dilations,
            hidden_dim=enccfg['hidden_dim'],
            depth=enccfg['depth'],
            heads=enccfg['heads'],
            pos_meb_dim=enccfg['pos_meb_dim'],
        )

        # 初始化 Codec Decoder
        deccfg = self.cfg.model.codec_decoder

        self.generator = CodecDecoderVocos(
            hidden_dim=deccfg.hidden_dim,     
            depth=deccfg.depth,
            heads=deccfg.heads,
            pos_meb_dim=deccfg.pos_meb_dim,
            hop_length=320,
            vq_num_quantizers=deccfg.vq_num_quantizers,  # VQ 量化器数量
            vq_dim=deccfg.vq_dim,                   # VQ 维度
            vq_commit_weight=deccfg.vq_commit_weight,    # VQ 提交权重
            vq_weight_init=deccfg.vq_weight_init,         # VQ 权重初始化
            vq_full_commit_loss=deccfg.vq_full_commit_loss,  # 是否使用完整的提交损失
            codebook_size=deccfg.codebook_size,            # 码本大小
            codebook_dim=deccfg.codebook_dim ,              # 码本维度
                  # 隐藏层维度
        )
        
 

        # 初始化 MultiPeriod Discriminator
        mpdcfg = self.cfg.model.mpd
        self.discriminator = HiFiGANMultiPeriodDiscriminator(
            periods=mpdcfg.periods,
            max_downsample_channels=mpdcfg.max_downsample_channels,
            channels=mpdcfg.channels,
            channel_increasing_factor=mpdcfg.channel_increasing_factor,
        )

        # 初始化 Spectral Discriminator
        mstftcfg = self.cfg.model.mstft
        self.spec_discriminator = SpecDiscriminator(
            stft_params=mstftcfg.stft_params,
            in_channels=mstftcfg.in_channels,
            out_channels=mstftcfg.out_channels,
            kernel_sizes=mstftcfg.kernel_sizes,
            channels=mstftcfg.channels,
            max_downsample_channels=mstftcfg.max_downsample_channels,
            downsample_scales=mstftcfg.downsample_scales,
            use_weight_norm=mstftcfg.use_weight_norm,
        )

 

        # 单独编译需要优化的子模块
        # self.CodecEnc = torch.compile(self.CodecEnc)
        # self.generator.backbone = torch.compile(self.generator )
        # self.mel_conv = torch.compile(self.mel_conv)
 
        self.model_spk = model_spk .eval()

        # self.semantic_model = AutoModel.from_pretrained("microsoft/wavlm-large")
        # self.semantic_model.eval()
        # self.semantic_model.requires_grad_(False)

 
        self.fc_prior = nn.Linear(1024 + 1024, deccfg.vq_dim,   )
        self.fc_post_a = nn.Linear(deccfg.vq_dim,  deccfg.hidden_dim )
        self.fc_post_s = nn.Linear(deccfg.vq_dim,   1024)

        self.SemanticDecoder_module = SemanticDecoder(1024, 1024, 1024)
        self.SemanticEncoder_module = SemanticEncoder(1024, 1024, 1024)
        self.semantic_model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0", output_hidden_states=True)
        self.semantic_model.eval()
        self.semantic_model.requires_grad_(False)
        # self.register_buffer('mel_basis', mel_basis)

        # self.perception_model = AutoModel.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        # self.perception_model.eval()
        # self.perception_model.requires_grad_(False)

        # ---------- Contrastive projection heads & hyperparams ----------
        # projection dim (可放到 cfg 中，若 cfg 中没有就用 256)
        try:
            proj_dim = self.cfg.model.proj_dim
        except Exception:
            proj_dim = 256

        # codec 投影 head（输入：vq_dim）
        try:
            vq_dim = deccfg.vq_dim
        except Exception:
            vq_dim = 256
        self.proj_codec = nn.Linear(vq_dim, proj_dim)

        # semantic 投影 head（输入：semantic dim，当前 semantic_model 输出是 1024）
        semantic_dim = 1024
        self.proj_sem = nn.Linear(semantic_dim, proj_dim)

        # 对比损失温度（cfg 中可配置）
        try:
            self.contrastive_temp = float(self.cfg.train.contrastive_temp)
        except Exception:
            self.contrastive_temp = 0.07

        # 初始化投影层（可选）
        nn.init.xavier_normal_(self.proj_codec.weight)
        nn.init.xavier_normal_(self.proj_sem.weight)


    def construct_criteria(self):
        cfg = self.cfg.train
        self.criteria = nn.ModuleDict()
        if cfg.use_mel_loss:
            self.criteria['mel_loss'] = MultiResolutionMelSpectrogramLoss(sample_rate=self.cfg.preprocess.audio.sr)
        if cfg.use_stft_loss:
            self.criteria['stft_loss'] = MultiResolutionSTFTLoss(
                fft_sizes=cfg.stft_loss_params.fft_sizes,
                hop_sizes=cfg.stft_loss_params.hop_sizes,
                win_sizes=cfg.stft_loss_params.win_lengths
            )
        if cfg.use_feat_match_loss:
            self.criteria['fm_loss'] = nn.L1Loss()
        self.criteria['gan_loss'] = GANLoss()
        self.criteria['l1_loss'] = nn.L1Loss()
        self.criteria['l2_loss'] = nn.MSELoss()
        print(self.criteria)





    def compute_audio_complexity(self, audio_features):
        """Compute the complexity of audio features (robust to AMP / half precision)."""
        # 修正变量名（之前是 autio_features）
        # spectral entropy
        spectral_entropy = self.compute_spectral_entropy(audio_features)
        # temporal variance：按最后一维求方差，然后取均值（根据你想要的聚合方式调整）
        temporal_variance = torch.var(audio_features, dim=-1).mean(dim=-1)
        complexity_score = (spectral_entropy + temporal_variance) / 2
        return complexity_score
    
    def compute_spectral_entropy(self, audio_features, pad_to_pow2=False):
        """
        Compute spectral entropy robustly:
          - If input is float16 (AMP), cast to float32 before FFT (cuFFT half has restriction).
          - Optionally pad last dim to next power-of-two if pad_to_pow2=True.
        Returns: entropy per-sample (or averaged)
        """
        # ensure float32 to avoid cuFFT half-precision restriction
        if audio_features.dtype == torch.float16:
            # cast to float32 (this keeps compute on GPU)
            audio_f = audio_features.float()
        else:
            audio_f = audio_features

        # optional: pad last dim to next power of two (uncomment if you prefer)
        if pad_to_pow2:
            n = audio_f.shape[-1]
            # next power of two
            n2 = 1 << (int(n - 1).bit_length())
            if n2 != n:
                pad_len = n2 - n
                # pad at the end
                audio_f = F.pad(audio_f, (0, pad_len))

        # compute FFT in float32 (safe)
        fft = torch.fft.fft(audio_f, dim=-1)
        power_spectrum = torch.abs(fft) ** 2
        # avoid division by zero
        denom = torch.sum(power_spectrum, dim=-1, keepdim=True).clamp_min(1e-12)
        prob_dist = power_spectrum / denom
        entropy = -torch.sum(prob_dist * torch.log(prob_dist + 1e-12), dim=-1)
        # 根据你的上游期望，返回每帧 entropy 的平均或其他聚合
        # 这里返回每样本的平均熵
        return entropy.mean(dim=-1)
 
 




    def forward(self, batch):
        wav = batch['wav']
        feats= batch['feats']
        
        vq_emb = self.CodecEnc(wav.unsqueeze(1))
        vq_emb = vq_emb.transpose(1, 2)

        complexity_score = self.compute_audio_complexity(vq_emb)

        with torch.no_grad():
            semantic_target = self.semantic_model(feats[:,0,:,:])

            semantic_target = semantic_target.hidden_states[16]
            semantic_target = semantic_target.detach()

        semantic_target = semantic_target.transpose(1, 2)
        semantic_target_processed = self.SemanticEncoder_module(semantic_target)
        # 拼接语义嵌入和编码器输出
        vq_emb = torch.cat([semantic_target_processed, vq_emb], dim=1)
        vq_emb = self.fc_prior(vq_emb.transpose(1, 2)).transpose(1, 2)

        vq_post_emb, vq_code, vq_loss = self.generator(vq_emb, vq=True, complexity_score=complexity_score) 
        semantic_recon = self.fc_post_s(vq_post_emb.transpose(1, 2)).transpose(1, 2)
        semantic_recon = self.SemanticDecoder_module(semantic_recon)

 
        y_ ,_ = self.generator(
            self.fc_post_a(vq_post_emb.transpose(1, 2)) ,
            vq=False
        )
        y = wav.unsqueeze(1)

        # gt_perceptual = self.perception_model(wav.squeeze(1), output_hidden_states=True) .hidden_states
        # gen_perceptual = self.perception_model(y_.squeeze(1), output_hidden_states=True) .hidden_states

        # gt_perceptual_se = gt_perceptual[10:22]
        # gen_perceptual_se = gen_perceptual[10:22]

        # perceptual_se_loss = [tensor1 - tensor2 for tensor1, tensor2 in zip(gt_perceptual_se, gen_perceptual_se)]

        # # 使用列表推导式逐元素相减
        # perceptual_se_loss_l2 = [F.mse_loss(tensor1.detach(), tensor2) for tensor1, tensor2 in zip(gt_perceptual_se, gen_perceptual_se)]
        # perceptual_se_loss_l2 =torch.stack(perceptual_se_loss_l2).mean()
        output = {
            'gt_wav': y,
            'gen_wav': y_,
            'vq_loss': vq_loss,
            'vq_code': vq_code,
            'semantic_recon_loss': F.mse_loss(semantic_recon, semantic_target),
            # 暴露中间表示用于 contrastive loss 计算
            'vq_post_emb': vq_post_emb,                      # [B, C_vq, T]
            'semantic_target_processed': semantic_target_processed,  # [B, C_sem, T_sem]
        }
        return output


    @torch.inference_mode()
    def inference(self, wav):
        vq_emb = self.CodecEnc(wav.unsqueeze(1))
        vq_post_emb, vq_code, vq_loss = self.generator(vq_emb, vq=True)
        y_ = self.generator(vq_post_emb, vq=False).squeeze(1)  # [B, T]
        return y_

    def compute_disc_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        y_ = y_.detach()
        p = self.discriminator(y)
        p_ = self.discriminator(y_)

        real_loss_list, fake_loss_list = [], []
        for i in range(len(p)):
            real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(p[i][-1], p_[i][-1])
            real_loss_list.append(real_loss)
            fake_loss_list.append(fake_loss)

        if hasattr(self, 'spec_discriminator'):
            sd_p = self.spec_discriminator(y)
            sd_p_ = self.spec_discriminator(y_)

            for i in range(len(sd_p)):
                real_loss, fake_loss = self.criteria['gan_loss'].disc_loss(sd_p[i][-1], sd_p_[i][-1])
                real_loss_list.append(real_loss)
                fake_loss_list.append(fake_loss)

        real_loss = sum(real_loss_list)
        fake_loss = sum(fake_loss_list)

        disc_loss = real_loss + fake_loss
        disc_loss = self.cfg.train.lambdas.lambda_disc * disc_loss

        output = {
            'real_loss': real_loss,
            'fake_loss': fake_loss,
            'disc_loss': disc_loss,
        }
        return output

    def compute_contrastive_loss(self, vq_post_emb, semantic_processed):
        """
        vq_post_emb: [B, C_vq, T1]
        semantic_processed: [B, C_sem, T2]
        返回: loss (scalar), info dict (proj_codec, proj_sem, logits)
        """

        # 平均池化（时间维度）
        # 如果 T1/T2 为 None 或 1，也能正常工作
        z_codec = vq_post_emb.mean(dim=-1)   # [B, C_vq]
        z_sem = semantic_processed.mean(dim=-1)  # [B, C_sem]

        # 线性投影到统一维度
        z_c = self.proj_codec(z_codec)   # [B, proj_dim]
        z_s = self.proj_sem(z_sem)       # [B, proj_dim]

        # L2 normalize
        z_c = F.normalize(z_c, p=2, dim=1)
        z_s = F.normalize(z_s, p=2, dim=1)

        B = z_c.size(0)
        # logits: z_c @ z_s^T / temp
        logits = torch.matmul(z_c, z_s.t()) / (self.contrastive_temp + 1e-12)  # [B, B]

        labels = torch.arange(B, device=logits.device, dtype=torch.long)
        # symmetric InfoNCE: loss(c->s) + loss(s->c)
        loss_c2s = F.cross_entropy(logits, labels)
        loss_s2c = F.cross_entropy(logits.t(), labels)
        loss = 0.5 * (loss_c2s + loss_s2c)

        info = {
            'contrastive_loss': loss,
            'loss_c2s': loss_c2s.detach(),
            'loss_s2c': loss_s2c.detach(),
            'logits_max': logits.detach().max(),
        }
        return loss, info


    def compute_gen_loss(self, batch, output):
        y, y_ = output['gt_wav'], output['gen_wav']
        vq_loss, vq_code = output['vq_loss'], output['vq_code']
        semantic_recon_loss = output['semantic_recon_loss']
        # perceptual_se_loss_l2 = output['perceptual_se_loss_l2']
        # x_feat_recon_loss = output['x_feat_recon_loss']
        gen_loss = 0.0
        self.set_discriminator_gradients(False)
        output_dict = {}
        cfg = self.cfg.train

        # Mel spectrogram loss
        if cfg.use_mel_loss:
            mel_loss = self.criteria['mel_loss'](y_.squeeze(1), y.squeeze(1))
            gen_loss += mel_loss * cfg.lambdas.lambda_mel_loss
            output_dict['mel_loss'] = mel_loss

        # GAN loss
        p_ = self.discriminator(y_)
        adv_loss_list = []
        for i in range(len(p_)):
            adv_loss_list.append(self.criteria['gan_loss'].gen_loss(p_[i][-1]))
        if hasattr(self, 'spec_discriminator'):
            sd_p_ = self.spec_discriminator(y_)
            for i in range(len(sd_p_)):
                adv_loss_list.append(self.criteria['gan_loss'].gen_loss(sd_p_[i][-1]))
        adv_loss = sum(adv_loss_list)
        gen_loss += adv_loss * cfg.lambdas.lambda_adv
        output_dict['adv_loss'] = adv_loss

        # Feature Matching loss
        if cfg.use_feat_match_loss:
            fm_loss = 0.0
            with torch.no_grad():
                p = self.discriminator(y)
            for i in range(len(p_)):
                for j in range(len(p_[i]) - 1):
                    fm_loss += self.criteria['fm_loss'](p_[i][j], p[i][j].detach())
            gen_loss += fm_loss * cfg.lambdas.lambda_feat_match_loss
            output_dict['fm_loss'] = fm_loss
            if hasattr(self, 'spec_discriminator'):
                spec_fm_loss = 0.0
                with torch.no_grad():
                    sd_p = self.spec_discriminator(y)
                for i in range(len(sd_p_)):
                    for j in range(len(sd_p_[i]) - 1):
                        spec_fm_loss += self.criteria['fm_loss'](sd_p_[i][j], sd_p[i][j].detach())
                gen_loss += spec_fm_loss * cfg.lambdas.lambda_feat_match_loss
                output_dict['spec_fm_loss'] = spec_fm_loss

        # VQ loss
        if vq_loss is not None:
            vq_loss = sum(vq_loss)
            gen_loss += vq_loss
            output_dict['vq_loss'] = vq_loss

        # Semantic reconstruction loss
                # Semantic reconstruction loss
        output_dict['semantic_recon_loss'] = semantic_recon_loss
        gen_loss += output_dict['semantic_recon_loss'] * cfg.lambdas.lambda_semantic_loss

        # ---------------- Contrastive semantic loss (InfoNCE) ----------------
        # 权重从 cfg 中读取（若 cfg 未配置就用 1.0）
        try:
            lambda_contrast = float(cfg.lambdas.lambda_contrast)
        except Exception:
            lambda_contrast = 1.0

        # 从 forward 返回的中间表示计算对比损失
        try:
            vq_post_emb = output['vq_post_emb']                    # [B, C_vq, T]
            semantic_processed = output['semantic_target_processed']  # [B, C_sem, T]
            contrastive_loss, contrast_info = self.compute_contrastive_loss(vq_post_emb, semantic_processed)
            gen_loss += contrastive_loss * lambda_contrast
            output_dict['contrastive_loss'] = contrastive_loss
            # 可记录更多信息（有助调参）
            output_dict['contrast_loss_c2s'] = contrast_info['loss_c2s']
            output_dict['contrast_loss_s2c'] = contrast_info['loss_s2c']
        except KeyError:
            # 若 forward 未返回需要的中间表示，则跳过
            output_dict['contrastive_loss'] = torch.tensor(0.0, device=gen_loss.device)


        # Perceptual loss
        # output_dict['perceptual_se_loss_l2'] = perceptual_se_loss_l2
        # gen_loss += output_dict['perceptual_se_loss_l2'] * cfg.lambdas.lambda_perceptual_loss
        
        self.set_discriminator_gradients(True)
        output_dict['gen_loss'] = gen_loss
        return output_dict

    def training_step(self, batch, batch_idx):
        output = self(batch)

        gen_opt, disc_opt = self.optimizers()
        gen_sche, disc_sche = self.lr_schedulers()

        # 训练判别器
        disc_losses = self.compute_disc_loss(batch, output)
        disc_loss = disc_losses['disc_loss']
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        self.clip_gradients(
            disc_opt,
            gradient_clip_val=self.cfg.train.disc_grad_clip,
            gradient_clip_algorithm='norm'
        )
        disc_opt.step()
        disc_sche.step()

        # 训练生成器
        gen_losses = self.compute_gen_loss(batch, output)
        gen_loss = gen_losses['gen_loss']
        gen_opt.zero_grad()
        self.manual_backward(gen_loss)
        self.clip_gradients(
            gen_opt,
            gradient_clip_val=self.cfg.train.gen_grad_clip,
            gradient_clip_algorithm='norm'
        )
        gen_opt.step()
        gen_sche.step()

        # 记录损失
        self.log_dict(
            disc_losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.cfg.dataset.train.batch_size,
            sync_dist=True
        )
        self.log_dict(
            gen_losses,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=self.cfg.dataset.train.batch_size,
            sync_dist=True
        )

    def validation_step(self, batch, batch_idx):
        # 您可以在此处实现验证逻辑
        output = self(batch)
        y = output['gt_wav']       # 真实音频
        y_ = output['gen_wav']  
           # 生成的重建音频
        embeddings1 = self.model_spk( y.squeeze(1))
        
        # 处理目标文件
        embeddings2 = self.model_spk(y_.squeeze(1))
        
        # 计算余弦相似度
        
        sim = F.cosine_similarity(embeddings1, embeddings2)
        sim = sim.mean()
        
        self.log('val/sim', sim, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return {'sim': sim}

 

    def test_step(self, batch, batch_idx):
        # 您可以在此处实现测试逻辑
        pass

    def configure_optimizers(self):
        from itertools import chain

        # 判别器参数
        disc_params = self.discriminator.parameters()
        # if hasattr(self, 'spec_discriminator'):
        disc_params = chain(disc_params, self.spec_discriminator.parameters())

        # 生成器参数
        gen_params = chain(
            self.CodecEnc.parameters(),
            self.generator.parameters(),
            # self.mel_conv.parameters(),
            self.fc_prior.parameters(),
            self.fc_post_a.parameters(),
            self.fc_post_s.parameters(),
            self.SemanticDecoder_module.parameters(),
            self.SemanticEncoder_module.parameters()
        )

        # 优化器
        gen_opt = optim.AdamW(gen_params, **self.cfg.train.gen_optim_params)
        disc_opt = optim.AdamW(disc_params, **self.cfg.train.disc_optim_params)

        # 学习率调度器
        gen_sche = WarmupLR(gen_opt, **self.cfg.train.gen_schedule_params)
        disc_sche = WarmupLR(disc_opt, **self.cfg.train.disc_schedule_params)

        print(f'Generator optim: {gen_opt}')
        print(f'Discriminator optim: {disc_opt}')

        return [gen_opt, disc_opt], [gen_sche, disc_sche]

    def set_discriminator_gradients(self, flag=True):
        for p in self.discriminator.parameters():
            p.requires_grad = flag

        if hasattr(self, 'spec_discriminator'):
            for p in self.spec_discriminator.parameters():
                p.requires_grad = flag
