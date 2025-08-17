import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import pytorch_lightning as pl
import random
import librosa
from os.path import basename, exists, join
from torch.utils.data import Dataset, DataLoader
import hydra
import utils
from transformers import AutoFeatureExtractor
from torchaudio.transforms import Resample
from tqdm import tqdm
import json

# ---------- 辅助：将 Windows 路径规范化为 Unix/WSL 路径 ----------
def _normalize_path_to_unix(p: str) -> str:
    """
    Convert Windows style path to unix (/mnt/<drive>/...) or normalize backslashes:
      - 'E:\\path\\to.wav' -> '/mnt/e/path/to.wav'
      - 'E:/path/to.wav'   -> '/mnt/e/path/to.wav'
      - already unix absolute -> unchanged
      - relative path with backslashes -> replace with '/'
    """
    if p is None:
        return p
    p = p.strip()
    if not p:
        return p
    # already unix abs path or file URI
    if p.startswith('/') or p.startswith('file://'):
        return p
    # drive letter pattern
    m = re.match(r'^([A-Za-z]):[\\/](.*)$', p)
    if m:
        drive = m.group(1).lower()
        rest = m.group(2).replace('\\', '/')
        return f"/mnt/{drive}/{rest}"
    # replace backslashes if present
    if '\\' in p:
        return p.replace('\\', '/')
    return p

class DataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        try:
            ocwd = hydra.utils.get_original_cwd()
        except Exception:
            ocwd = os.getcwd()
        self.ocwd = ocwd

    def get_loader(self, phase):
        phase_cfg = self.cfg.get(phase)  # 修改为顶级键
        batch_size = phase_cfg.batch_size
        num_workers = min(os.cpu_count() // 2, 8)
        ds = FSDataset_add_STFT(phase, self.cfg)  # 使用 FSDataset_add_STFT
        dl = DataLoader(ds,
                        batch_size=batch_size,
                        shuffle=phase_cfg.shuffle,
                        num_workers=num_workers,
                        collate_fn=ds.collate_fn,
                        pin_memory=True,
                        persistent_workers=True)
        return dl

    def train_dataloader(self):
        return self.get_loader('train')

    def val_dataloader(self):
        return self.get_loader('val')

    def test_dataloader(self):
        return self.get_loader('test')

class FSDataset_add_STFT(Dataset):
    """Dataset batching wav, mel and other acoustic features with STFT"""
    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.get(phase)  # 使用顶级键
        try:
            self.ocwd = hydra.utils.get_original_cwd()
        except Exception:
            self.ocwd = os.getcwd()

        self.sr = cfg.preprocess.audio.sr
        self.filelist = self.get_filelist(self.phase_cfg.filelist)
        self.min_audio_length = cfg.dataset.min_audio_length
        # self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            "facebook/w2v-bert-2.0",
            cache_dir="/workspace/xcodec2n/models/huggingface"
        )

    def get_filelist(self, fpath):
        """Read JSON manifest and extract audio paths."""
        flist = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    wav = item.get("audio_filepath", "")
                except json.JSONDecodeError:
                    wav = line.split('\t')[0]
                wav = _normalize_path_to_unix(wav)
                flist.append(wav)
        return flist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, idx):
        wavpath = self.filelist[idx]
        wavpath = _normalize_path_to_unix(wavpath)

        if os.path.isabs(wavpath):
            wavpath_full = wavpath
        else:
            root = _normalize_path_to_unix(self.cfg.preprocess.datasets.LibriSpeech.root)
            if root:
                wavpath_full = os.path.join(root, wavpath)
            else:
                wavpath_full = wavpath

        wav, sr = torchaudio.load(wavpath_full)
        if sr != 16000:
            wav = Resample(sr, 16000)(wav)
        wav = wav[0, :]
        length = wav.shape[0]

        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[0]

        i = random.randint(0, length - self.min_audio_length)
        wav = wav[i:i + self.min_audio_length]

        # 添加 STFT 特征（示例）
        stft = torch.stft(wav, n_fft=512, hop_length=256, return_complex=True)
        stft = torch.abs(stft)

        wav_pad = F.pad(wav, (160, 160))
        feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt").data['input_features']

        out = {
            'wav': wav,
            'feat': feat,
            'stft': stft
        }
        return out

    def collate_fn(self, bs):
        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        feats = [b['feat'] for b in bs]
        feats = torch.stack(feats)
        stfts = [b['stft'] for b in bs]
        stfts = torch.stack(stfts)
        out = {
            'wav': wavs,
            'feats': feats,
            'stfts': stfts
        }
        return out

@hydra.main(config_path='config', config_name='default', version_base='1.1')
def main(cfg):
    data_module = DataModule(cfg)
    train_loader = data_module.train_dataloader()  # 修改为 train_dataloader
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches", unit="batch")):
        print(f"[DEBUG] Batch wav shape: {batch['wav'].shape}")
        print(f"[DEBUG] Batch feats shape: {batch['feats'].shape}")
        print(f"[DEBUG] Batch stfts shape: {batch['stfts'].shape}")
        break

if __name__ == "__main__":
    main()