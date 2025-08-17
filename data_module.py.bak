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
        phase_cfg = self.cfg.dataset.get(phase)
        batch_size = phase_cfg.batch_size
        # allow overriding num_workers in config, default to 8
        # num_workers = int(self.cfg.dataset.get('num_workers', 4))
        num_workers = min(os.cpu_count() // 2, 8)
        ds = FSDataset(phase, self.cfg)
        # ds = FSDataset_add_STFT(phase, self.cfg)
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

class FSDataset(Dataset):
    """Dataset batching wav, mel and other acoustic features"""

    def __init__(self, phase, cfg):
        self.phase = phase
        self.cfg = cfg
        self.phase_cfg = cfg.dataset.get(phase)
        try:
            self.ocwd = hydra.utils.get_original_cwd()
        except Exception:
            self.ocwd = os.getcwd()

        self.sr = cfg.preprocess.audio.sr
        # load and normalize filelist
        self.filelist = self.get_filelist(self.phase_cfg.filelist)
        self.min_audio_length = cfg.dataset.min_audio_length
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")

    def __len__(self):
        return len(self.filelist)

    def load_wav(self, path):
        wav, sr = librosa.load(path, sr=self.sr)
        return wav

    # def get_filelist(self, fpath):
    #     """Read filelist and normalize paths; supports tab-separated extra columns."""
    #     flist = []
    #     with open(fpath, 'r', encoding='utf-8') as f:
    #         for line in f:
    #             line = line.strip()
    #             if not line:
    #                 continue
    #             wav = line.split('\t')[0]
    #             wav = _normalize_path_to_unix(wav)
    #             flist.append(wav)
    #     return flist
    def get_filelist(self, fpath):
        """Read JSON manifest and extract audio paths."""
        flist = []
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    # 如果是 JSON 格式
                    item = json.loads(line)
                    wav = item.get("audio_filepath", "")
                except json.JSONDecodeError:
                    # 如果不是 JSON，就按原逻辑走
                    wav = line.split('\t')[0]
                wav = _normalize_path_to_unix(wav)
                flist.append(wav)
        return flist

    def __getitem__(self, idx):
        wavpath = self.filelist[idx]
        # normalize again just in case
        wavpath = _normalize_path_to_unix(wavpath)

        # If path is absolute (unix style), use directly; otherwise join with config root
        if os.path.isabs(wavpath):
            wavpath_full = wavpath
        else:
            # normalize root in case it's Windows style in config
            root = _normalize_path_to_unix(self.cfg.preprocess.datasets.LibriSpeech.root)
            if root:
                wavpath_full = os.path.join(root, wavpath)
            else:
                wavpath_full = wavpath

        # DEBUG: 若第一次运行需要检查路径，把下面注释去掉
        # print(f"[DATA] idx={idx} wavpath={wavpath} full={wavpath_full} exists={os.path.exists(wavpath_full)}")

        # load audio
        wav, sr = torchaudio.load(wavpath_full)

        # resample if needed and convert to 1D tensor
        if sr != 16000:
            wav = Resample(sr, 16000)(wav)
        wav = wav[0, :]
        length = wav.shape[0]

        if length < self.min_audio_length:
            wav = F.pad(wav, (0, self.min_audio_length - length))
            length = wav.shape[0]

        i = random.randint(0, length - self.min_audio_length)
        wav = wav[i:i + self.min_audio_length]

        wav_pad = F.pad(wav, (160, 160))
        feat = self.feature_extractor(wav_pad, sampling_rate=16000, return_tensors="pt").data['input_features']

        out = {
            'wav': wav,
            'feat': feat,
        }
        return out

    def collate_fn(self, bs):
        wavs = [b['wav'] for b in bs]
        wavs = torch.stack(wavs)
        feats = [b['feat'] for b in bs]
        feats = torch.stack(feats)
        out = {
            'wav': wavs,
            'feats': feats,
        }
        return out

@hydra.main(config_path='config', config_name='default')
def main(cfg):
    data_module = DataModule(cfg)
    train_loader = data_module.val_dataloader()
    for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing batches", unit="batch")):
        wavs = batch['wav']

if __name__ == "__main__":
    main()
