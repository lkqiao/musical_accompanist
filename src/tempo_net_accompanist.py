#!/usr/bin/env python3
# tempo_net_accompanist.py
# End-to-end ML for per-second tempo (BPM) estimation using accompanist pseudo-labels.
# Usage:
#   Train:
#     python tempo_net_accompanist.py train \
#       --csv train.csv \
#       --accomp_config_path /path/to/config.yaml \
#       --out_dir runs/tempo_net \
#       --epochs 60 --device cuda
#     # train.csv columns: audio_path[,name]
#
#   Infer:
#     python tempo_net_accompanist.py infer \
#       --audio path/to/song.wav \
#       --checkpoint runs/tempo_net/ckpt_best.pt \
#       --out_csv preds.csv --device cpu
#
# Author: Luke Qiao

import os, math, argparse, random
from dataclasses import dataclass
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import soundfile as sf
from scipy.ndimage import gaussian_filter1d

# teacher model
import accompanist

# Utilities
def set_seed(seed=1337):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def logspace_tempo_bins(bpm_min=30.0, bpm_max=240.0, K=100):
    return np.geomspace(bpm_min, bpm_max, K)

def soft_labels_for_bpm(bpm, bins, sigma_bins=1.5, include_octaves=True):
    """
    Soft label around BPM (and optionally x0.5/x2) in log-tempo bin space.
    """
    logb = np.log(bins)
    targets = np.zeros_like(bins, dtype=np.float32)
    centers = [bpm]
    if include_octaves:
        centers += [bpm/2.0, bpm*2.0]
    step = np.mean(np.diff(logb))
    for c in centers:
        if not np.isfinite(c) or c <= 0: continue
        z = (logb - np.log(c)) / (step + 1e-12)
        targets += np.exp(-0.5 * (z / sigma_bins) ** 2)
    targets = targets + 1e-8
    targets /= targets.sum()
    return targets

def viterbi_decode(logits_TK, trans_penalty=2.0):
    """
    Viterbi across seconds with distance-based transition cost.
    logits_TK: (T,K) log-probs or logits.
    """
    T, K = logits_TK.shape
    j = np.arange(K)[None, :]
    i = np.arange(K)[:, None]
    trans = -trans_penalty * np.abs(i - j)         # (K,K)
    dp = np.full((T, K), -1e18, dtype=np.float32)
    ptr = np.zeros((T, K), dtype=np.int32)
    dp[0] = logits_TK[0]
    for t in range(1, T):
        scores = dp[t-1][:, None] + trans         # (K,K)
        ptr[t] = np.argmax(scores, axis=0)
        dp[t]  = logits_TK[t] + scores[ptr[t], np.arange(K)]
    path = np.zeros(T, dtype=np.int32)
    path[-1] = np.argmax(dp[-1])
    for t in range(T-2, -1, -1):
        path[t] = ptr[t+1, path[t+1]]
    return path

# Feature extraction
@dataclass
class FeatConfig:
    sr: int = 22050
    n_fft: int = 2048
    hop_length: int = 256     # ~11.6ms at 22.05k
    n_mels: int = 128
    fmin: float = 20
    fmax: float = 9000
    context_sec: float = 8.0  # model window length (seconds)
    out_hop_sec: float = 1.0  # output each 1s

def extract_features(y, sr, cfg: FeatConfig):
    """
    Returns:
      S:   (n_mels, T)   log-mel
      Tgm: (n_tmp,  T)   Fourier tempogram from onset envelope
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=cfg.n_fft, hop_length=cfg.hop_length,
        n_mels=cfg.n_mels, fmin=cfg.fmin, fmax=cfg.fmax, power=2.0
    )
    S = np.log(1e-6 + S).astype(np.float32)

    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length, aggregate=np.median)
    win_length = int(round(cfg.context_sec * sr / cfg.hop_length))
    Tgm = librosa.feature.tempogram(
        onset_envelope=onset_env, sr=sr, hop_length=cfg.hop_length, win_length=win_length
    ).astype(np.float32)

    T = min(S.shape[1], Tgm.shape[1])
    return S[:, :T], Tgm[:, :T]

def frame_indices_for_seconds(sr, hop, total_frames, context_sec=8.0, out_hop_sec=1.0):
    """
    Returns list of (start_frame, end_frame, center_sec) for each 1s step,
    using a centered 'context_sec' window.
    """
    fps = sr / hop
    half = int(round((context_sec/2) * fps))
    step = int(round(out_hop_sec * fps))
    centers = np.arange(half, total_frames - half, step)
    return [(c - half, c + half, c / fps) for c in centers]

# Model
class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.BatchNorm2d(c_out),
            nn.ReLU(),
            nn.MaxPool2d((2,2))
        )
    def forward(self, x): return self.seq(x)

class TCN(nn.Module):
    def __init__(self, c_in, h=256, layers=4, k=3):
        super().__init__()
        layers_list=[]
        c = c_in
        d=1
        for _ in range(layers):
            layers_list += [
                nn.Conv1d(c, h, k, padding=d*(k-1)//2, dilation=d),
                nn.ReLU(),
                nn.Conv1d(h, c, k, padding=d*(k-1)//2, dilation=d),
                nn.ReLU()
            ]
            d *= 2
        self.net = nn.Sequential(*layers_list)
    def forward(self, x): return self.net(x)  # B,C,T

class TempoNet(nn.Module):
    def __init__(self, n_mel=128, n_tmp=96, n_bins=100):
        super().__init__()
        # mel branch
        self.m1 = ConvBlock(1, 32); self.m2 = ConvBlock(32, 64)
        # rhythm branch
        self.r1 = ConvBlock(1, 32); self.r2 = ConvBlock(32, 64)
        # fusion
        feat_m = 64 * (n_mel//4)
        feat_r = 64 * (n_tmp//4)
        self.proj = nn.Linear(feat_m + feat_r, 256)
        self.tcn  = TCN(256, h=256, layers=4, k=3)
        self.head = nn.Conv1d(256, n_bins, 1)  # logits over tempo bins

    def forward(self, mel, tmp):
        # mel/tmp: B,1,F,Twin
        B,_,Fm,Tw = mel.shape
        _,_,Fr,_ = tmp.shape

        m = self.m2(self.m1(mel))            # B,64,Fm/4,T'
        r = self.r2(self.r1(tmp))            # B,64,Fr/4,T'

        m = m.permute(0,3,1,2).reshape(B, -1, m.shape[3])  # B,Cm,T'
        r = r.permute(0,3,1,2).reshape(B, -1, r.shape[3])  # B,Cr,T'
        Tprime = min(m.shape[-1], r.shape[-1])
        m = m[..., :Tprime]; r = r[..., :Tprime]
        x = torch.cat([m, r], dim=1)         # B, Cm+Cr, T'
        x = x.permute(0,2,1)                 # B, T', C
        x = F.relu(self.proj(x))             # B, T', 256
        x = x.permute(0,2,1)                 # B, 256, T'
        x = self.tcn(x)                      # B, 256, T'
        logits = self.head(x)                # B, K, T'
        return logits

# Labels from accompanist
def labels_from_accompanist(y, sr, bins, name, config_path, max_wait_time, n_seconds):
    """
    Calls accompanist.estimate_tempo(signal, sr, name, config_path, max_wait_time)
    and returns a (n_seconds, K) array of soft labels at 1 Hz.
    Accepted return formats:
      - single BPM (float / int)
      - dict {'time': [...], 'bpm': [...]} (seconds, BPM)
      - tuple/list (times, bpms)
      - 1D array of BPM sampled at 1 Hz (assumed)
    """
    est = accompanist.estimate_tempo(
        signal=y, sr=sr, name=name, config_path=config_path, max_wait_time=max_wait_time
    )

    # Case A: single number
    if isinstance(est, (int, float, np.floating)):
        bpm_val = float(est)
        return np.stack([soft_labels_for_bpm(bpm_val, bins, include_octaves=True)
                         for _ in range(n_seconds)], axis=0)

    # Parse common time-series shapes
    if isinstance(est, dict) and 'time' in est and 'bpm' in est:
        times = np.asarray(est['time'], float)
        bpms  = np.asarray(est['bpm'],  float)
    elif isinstance(est, (tuple, list)) and len(est) == 2:
        times = np.asarray(est[0], float)
        bpms  = np.asarray(est[1], float)
    else:
        # assume 1 Hz BPM sequence starting at 0.5s centers
        bpms  = np.asarray(est, float)
        times = np.arange(len(bpms), dtype=float) + 0.5

    centers = np.arange(n_seconds, dtype=float) + 0.5
    bpm_sec = np.interp(centers, times, bpms, left=bpms[0], right=bpms[-1])
    return np.stack([soft_labels_for_bpm(float(b), bins, include_octaves=True) for b in bpm_sec], axis=0)

# Dataset
class TempoDataset(Dataset):
    """
    Produces training windows and soft labels from accompanist pseudo-labels.
    CSV columns: audio_path[,name]
    """
    def __init__(self, csv_path, cfg: FeatConfig,
                 bpm_min=30.0, bpm_max=240.0, n_bins=100,
                 accomp_config_path=None, accomp_max_wait_time=30.0,
                 aug_time_stretch=0.08, train=True):
        self.df = pd.read_csv(csv_path)
        self.cfg = cfg
        self.train = train
        self.bins = logspace_tempo_bins(bpm_min, bpm_max, n_bins)
        self.accomp_config_path = accomp_config_path
        self.accomp_max_wait_time = float(accomp_max_wait_time)
        self.aug_time_stretch = aug_time_stretch

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        wav = str(row['audio_path'])
        name = str(row['name']) if 'name' in row and pd.notna(row['name']) else os.path.basename(wav)

        y, sr = librosa.load(wav, sr=self.cfg.sr, mono=True)

        # Light augmentation (train only). We stretch audio; accompanist sees stretched audio (no label correction needed).
        if self.train and self.aug_time_stretch > 0:
            factor = 1.0 + np.random.uniform(-self.aug_time_stretch, self.aug_time_stretch)
            y = librosa.effects.time_stretch(y, factor)

        S, Tgm = extract_features(y, sr, self.cfg)
        spans = frame_indices_for_seconds(self.cfg.sr, self.cfg.hop_length, S.shape[1],
                                          self.cfg.context_sec, self.cfg.out_hop_sec)
        out_T = len(spans)
        if out_T == 0:
            # too short, pad center
            need_frames = int(round(self.cfg.context_sec * (sr/self.cfg.hop_length)))
            if S.shape[1] < need_frames:
                pad = need_frames - S.shape[1]
                S = np.pad(S, ((0,0),(pad//2, pad - pad//2)), mode='edge')
                Tgm = np.pad(Tgm, ((0,0),(pad//2, pad - pad//2)), mode='edge')
                spans = frame_indices_for_seconds(self.cfg.sr, self.cfg.hop_length, S.shape[1],
                                                  self.cfg.context_sec, self.cfg.out_hop_sec)
                out_T = len(spans)
            if out_T == 0:
                # still no spans, fallback one window at middle
                mid = S.shape[1]//2
                half = int(round((self.cfg.context_sec/2) * (sr/self.cfg.hop_length)))
                spans = [(mid-half, mid+half, (mid)/(sr/self.cfg.hop_length))]
                out_T = 1

        # Build labels from accompanist (once per file)
        Y_soft = labels_from_accompanist(
            y=y, sr=sr, bins=self.bins, name=name,
            config_path=self.accomp_config_path, max_wait_time=self.accomp_max_wait_time,
            n_seconds=out_T
        )  # (T, K)

        # Assemble windows
        Xmel, Xtmp, Y = [], [], []
        win = spans[0][1] - spans[0][0]
        for t, (a,b,_) in enumerate(spans):
            mel_win = S[:, a:b]
            tmp_win = Tgm[:, a:b]
            if mel_win.shape[1] != win or tmp_win.shape[1] != win:
                continue
            Xmel.append(mel_win[None, ...])   # add channel
            Xtmp.append(tmp_win[None, ...])
            Y.append(Y_soft[t])

        if len(Xmel) == 0:
            # fallback single sample
            mid = S.shape[1]//2
            a = mid - win//2; b = a + win
            Xmel = [S[:, a:b][None, ...]]
            Xtmp = [Tgm[:, a:b][None, ...]]
            Y = [soft_labels_for_bpm(120.0, self.bins)]

        Xmel = torch.from_numpy(np.stack(Xmel)).float()  # N,1,F,T
        Xtmp = torch.from_numpy(np.stack(Xtmp)).float()
        Y    = torch.from_numpy(np.stack(Y)).float()     # N,K
        return Xmel, Xtmp, Y

def collate_concat(batch):
    Xmels = torch.cat([x[0] for x in batch], dim=0)
    Xtmps = torch.cat([x[1] for x in batch], dim=0)
    Ys    = torch.cat([x[2] for x in batch], dim=0)
    return Xmels, Xtmps, Ys

# Training
def train_loop(args):
    set_seed(args.seed)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')

    fcfg = FeatConfig(sr=args.sr, n_fft=args.n_fft, hop_length=args.hop_length,
                      n_mels=args.n_mels, fmin=args.fmin, fmax=args.fmax,
                      context_sec=args.context_sec, out_hop_sec=1.0)

    ds = TempoDataset(args.csv, fcfg,
                      bpm_min=args.bpm_min, bpm_max=args.bpm_max, n_bins=args.n_bins,
                      accomp_config_path=args.accomp_config_path,
                      accomp_max_wait_time=args.max_wait_time,
                      aug_time_stretch=0.08, train=True)

    # 90/10 split (quick baseline)
    n = len(ds); n_val = max(1, int(0.1*n))
    idx = np.arange(n); np.random.shuffle(idx)
    val_idx = idx[:n_val]; tr_idx = idx[n_val:]
    from torch.utils.data import Subset
    tr = Subset(ds, tr_idx); va = Subset(ds, val_idx)

    dl_tr = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                       collate_fn=collate_concat, pin_memory=True)
    dl_va = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                       collate_fn=collate_concat, pin_memory=True)

    # probe n_tmp
    tmp_sample = ds[0][1]
    n_tmp = tmp_sample.shape[2]
    model = TempoNet(n_mel=args.n_mels, n_tmp=n_tmp, n_bins=args.n_bins).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', patience=4, factor=0.5)
    best_val = 1e9

    def loss_fn(logits, targ):
        # Pool over internal time dim -> B,K; KL to soft labels.
        logp = F.log_softmax(logits.mean(dim=-1), dim=1)  # B,K
        return F.kl_div(logp, targ, reduction='batchmean')

    for epoch in range(1, args.epochs+1):
        model.train(); tr_loss=0
        for Xm, Xt, Y in tqdm(dl_tr, desc=f"epoch {epoch} train"):
            Xm, Xt, Y = Xm.to(device), Xt.to(device), Y.to(device)
            logits = model(Xm, Xt)   # B,K,T'
            loss = loss_fn(logits, Y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item() * Xm.shape[0]
        tr_loss /= len(tr)

        model.eval(); va_loss=0
        with torch.no_grad():
            for Xm, Xt, Y in tqdm(dl_va, desc=f"epoch {epoch} val"):
                Xm, Xt, Y = Xm.to(device), Xt.to(device), Y.to(device)
                logits = model(Xm, Xt)
                loss = loss_fn(logits, Y)
                va_loss += loss.item() * Xm.shape[0]
        va_loss /= len(va)
        scheduler.step(va_loss)
        print(f"[epoch {epoch}] train {tr_loss:.4f}  val {va_loss:.4f}")

        ckpt = {
            'model': model.state_dict(),
            'bins': logspace_tempo_bins(args.bpm_min, args.bpm_max, args.n_bins).tolist(),
            'feat_cfg': fcfg.__dict__,
            'args': vars(args)
        }
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save(ckpt, os.path.join(args.out_dir, 'ckpt_last.pt'))
        if va_loss < best_val:
            best_val = va_loss
            torch.save(ckpt, os.path.join(args.out_dir, 'ckpt_best.pt'))
            print(f"  saved ckpt_best.pt (val {va_loss:.4f})")

# Inference
def infer_file(args):
    set_seed(0)
    device = torch.device(args.device if (args.device != 'cuda' or torch.cuda.is_available()) else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    bins = np.array(ckpt['bins'], dtype=float)
    fcfg = FeatConfig(**ckpt['feat_cfg'])

    y, sr = librosa.load(args.audio, sr=fcfg.sr, mono=True)
    S, Tgm = extract_features(y, sr, fcfg)
    spans = frame_indices_for_seconds(fcfg.sr, fcfg.hop_length, S.shape[1], fcfg.context_sec, 1.0)

    n_tmp = Tgm.shape[0]
    model = TempoNet(n_mel=fcfg.n_mels, n_tmp=n_tmp, n_bins=len(bins))
    model.load_state_dict(ckpt['model'])
    model.to(device).eval()

    Xm, Xt, centers = [], [], []
    win = spans[0][1] - spans[0][0]
    for (a,b,csec) in spans:
        mel_win = S[:, a:b]; tmp_win = Tgm[:, a:b]
        if mel_win.shape[1] != win or tmp_win.shape[1] != win: continue
        Xm.append(mel_win[None,...]); Xt.append(tmp_win[None,...]); centers.append(csec)

    if len(Xm) == 0:
        print("No windows produced; audio too short?")
        return

    Xm = torch.from_numpy(np.stack(Xm)).float().to(device)
    Xt = torch.from_numpy(np.stack(Xt)).float().to(device)

    with torch.no_grad():
        logits = model(Xm, Xt).mean(dim=-1)          # N,K
        logp = F.log_softmax(logits, dim=1).cpu().numpy()

    path_idx = viterbi_decode(logp, trans_penalty=args.trans_penalty)  # (N,)
    bpm_seq = bins[path_idx]

    if args.kalman_like_sigma > 0:
        ylog = np.log(np.clip(bpm_seq, 1e-6, None))
        ylog = gaussian_filter1d(ylog, sigma=args.kalman_like_sigma)
        bpm_seq = np.exp(ylog)

    df = pd.DataFrame({'time_sec': np.round(centers, 3), 'bpm_pred': np.round(bpm_seq, 3)})
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} with {len(df)} rows")

# CLI
def build_argparser():
    p = argparse.ArgumentParser(description="Per-second tempo estimation with accompanist pseudo-labels.")
    sub = p.add_subparsers(dest='cmd', required=True)

    pt = sub.add_parser('train', help='Train model with accompanist labels')
    pt.add_argument('--csv', required=True, help='CSV with columns: audio_path[,name]')
    pt.add_argument('--out_dir', default='runs/tempo_net')
    # features
    pt.add_argument('--sr', type=int, default=22050)
    pt.add_argument('--n_fft', type=int, default=2048)
    pt.add_argument('--hop_length', type=int, default=256)
    pt.add_argument('--n_mels', type=int, default=128)
    pt.add_argument('--fmin', type=float, default=20)
    pt.add_argument('--fmax', type=float, default=9000)
    pt.add_argument('--context_sec', type=float, default=8.0)
    # tempo bins
    pt.add_argument('--bpm_min', type=float, default=30.0)
    pt.add_argument('--bpm_max', type=float, default=240.0)
    pt.add_argument('--n_bins', type=int, default=100)
    # training
    pt.add_argument('--batch_size', type=int, default=6)
    pt.add_argument('--epochs', type=int, default=60)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--workers', type=int, default=2)
    pt.add_argument('--device', default='cuda')
    pt.add_argument('--seed', type=int, default=1337)
    # accompanist
    pt.add_argument('--accomp_config_path', required=True, help='Path to accompanist config file')
    pt.add_argument('--max_wait_time', type=float, default=30.0)

    pi = sub.add_parser('infer', help='Run inference on one audio file')
    pi.add_argument('--audio', required=True)
    pi.add_argument('--checkpoint', required=True)
    pi.add_argument('--out_csv', required=True)
    pi.add_argument('--device', default='cpu')
    pi.add_argument('--trans_penalty', type=float, default=2.0, help='Viterbi bin-jump penalty (higher=smoother)')
    pi.add_argument('--kalman_like_sigma', type=float, default=0.8, help='Gaussian smoothing sigma (seconds) in log-BPM')

    return p

def main():
    args = build_argparser().parse_args()
    if args.cmd == 'train':
        train_loop(args)
    elif args.cmd == 'infer':
        infer_file(args)
    else:
        raise ValueError("Unknown cmd")

if __name__ == '__main__':
    main()