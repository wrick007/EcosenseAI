"""
audio.py — audio loading, log-mel conversion, and acoustic-index features.

Returns both:
- clips: (1, 6, 1, 128, 216)
- idxs : (1, 6, 4), normalized [ACI, H, NDSI, BI]
"""

import json
import math
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T

SR = 22050
CLIP_SEC = 5
SEQ_LEN = 6
N_MELS = 128
HOP = 512
NFFT = 2048
IMG_W = 216
IDX_COLS = ["ACI", "H", "NDSI", "BI"]

_mel = T.MelSpectrogram(
    sample_rate=SR,
    n_fft=NFFT,
    hop_length=HOP,
    n_mels=N_MELS,
    f_min=50,
    f_max=11025,
)
_db = T.AmplitudeToDB(stype="power", top_db=80)


def _load(path: str) -> torch.Tensor:
    wav_np, _ = librosa.load(path, sr=SR, mono=True)
    wav = torch.tensor(wav_np, dtype=torch.float32)
    mx = wav.abs().max()
    if mx > 0:
        wav = wav / mx
    return wav


def _to_logmel(wav: torch.Tensor) -> torch.Tensor:
    m = _db(_mel(wav.unsqueeze(0)))
    m = (m - m.mean()) / (m.std() + 1e-8)
    if m.shape[-1] < IMG_W:
        m = F.pad(m, (0, IMG_W - m.shape[-1]))
    return m[..., :IMG_W]


def acoustic_indices_from_wav(wav_np: np.ndarray) -> Dict[str, float]:
    S = np.abs(librosa.stft(wav_np, n_fft=NFFT, hop_length=HOP))
    freq = librosa.fft_frequencies(sr=SR, n_fft=NFFT)

    aci = float((np.abs(np.diff(S, axis=1)) / (S[:, :-1] + 1e-8)).sum())

    env = S.sum(0)
    env = env / (env.sum() + 1e-8)
    sp = S.mean(1)
    sp = sp / (sp.sum() + 1e-8)
    h = float(-np.sum(env * np.log(env + 1e-12)) * -np.sum(sp * np.log(sp + 1e-12)))

    am = S[(freq >= 1000) & (freq < 2000), :].sum()
    bio = S[(freq >= 2000) & (freq <= 8000), :].sum()
    ndsi = float((bio - am) / (bio + am + 1e-8))

    bio_band = S[(freq >= 2000) & (freq <= 8000), :]
    if bio_band.size == 0 or np.max(bio_band) <= 0:
        bi = 0.0
    else:
        bi = float(librosa.amplitude_to_db(bio_band, ref=np.max).mean())

    return {"ACI": aci, "H": h, "NDSI": ndsi, "BI": bi}


def acoustic_indices(path: str) -> Dict[str, float]:
    wav = _load(path)
    idx = acoustic_indices_from_wav(wav.detach().cpu().numpy())
    return {
        "ACI": round(idx["ACI"], 1),
        "H": round(idx["H"], 4),
        "NDSI": round(idx["NDSI"], 4),
        "BI": round(idx["BI"], 2),
    }


def load_idx_stats(path: str = "idx_stats.json") -> Dict[str, Dict[str, float]]:
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(
            "Missing idx_stats.json. Create it from the Colab training manifest using "
            "tools/export_idx_stats.py, then place it beside app.py."
        )
    return json.loads(stats_path.read_text())


def normalize_indices(raw: Dict[str, float], stats: Dict[str, Dict[str, float]]) -> torch.Tensor:
    vals = []
    for col in IDX_COLS:
        mn = float(stats[col]["min"])
        mx = float(stats[col]["max"])
        vals.append((float(raw[col]) - mn) / (mx - mn + 1e-8))
    return torch.tensor(vals, dtype=torch.float32)


def load_audio(path: str, idx_stats_path: str = "idx_stats.json") -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, Dict[str, float]]:
    stats = load_idx_stats(idx_stats_path)
    wav = _load(path)
    tgt = SR * CLIP_SEC

    if len(wav) < tgt * SEQ_LEN:
        wav = wav.repeat(math.ceil(tgt * SEQ_LEN / len(wav)))
    wav = wav[: tgt * SEQ_LEN]

    clips = []
    idxs = []
    raw_clip_indices = []

    for i in range(SEQ_LEN):
        seg = wav[i * tgt : (i + 1) * tgt]
        clips.append(_to_logmel(seg))
        raw = acoustic_indices_from_wav(seg.detach().cpu().numpy())
        raw_clip_indices.append(raw)
        idxs.append(normalize_indices(raw, stats))

    clips_tensor = torch.stack(clips).unsqueeze(0)
    idxs_tensor = torch.stack(idxs).unsqueeze(0)
    mel_full = _db(_mel(wav.unsqueeze(0))).squeeze(0).detach().cpu().numpy()

    avg_raw = {
        col: round(float(np.mean([r[col] for r in raw_clip_indices])), 4)
        for col in IDX_COLS
    }
    avg_raw["ACI"] = round(avg_raw["ACI"], 1)
    avg_raw["BI"] = round(avg_raw["BI"], 2)

    return clips_tensor, idxs_tensor, mel_full, avg_raw
