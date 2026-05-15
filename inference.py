import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import streamlit as st

from model import EcoSenseModel

WEIGHTS_PATH = Path("models/best.pt")


def get_hf_repo():
    try:
        return st.secrets.get("HF_REPO", os.environ.get("HF_REPO", ""))
    except FileNotFoundError:
        return os.environ.get("HF_REPO", "")


def load_model():
    if not WEIGHTS_PATH.exists():
        _download_weights()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = EcoSenseModel(d=384, seq_len=6, n_idx=4).to(device)

    state = torch.load(WEIGHTS_PATH, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    clean_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k.replace("module.", "", 1)
        clean_state[k] = v

    model.load_state_dict(clean_state, strict=True)
    model.eval()

    return model


def _download_weights():
    HF_REPO = get_hf_repo()

    if not HF_REPO:
        st.error(
            "Model weights not found. Put your checkpoint at `models/best.pt`."
        )
        st.stop()

    try:
        from huggingface_hub import hf_hub_download

        with st.spinner("Downloading model weights..."):
            WEIGHTS_PATH.parent.mkdir(exist_ok=True)

            hf_hub_download(
                repo_id=HF_REPO,
                filename="best.pt",
                local_dir=str(WEIGHTS_PATH.parent),
                local_dir_use_symlinks=False,
            )

    except Exception as e:
        st.error(f"Download failed: {e}")
        st.stop()


def _enable_dropout(model):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def _status(score):
    if score > 0.75:
        return "Pristine ecosystem"
    elif score > 0.55:
        return "Recovering — moderate biodiversity"
    elif score > 0.35:
        return "Degraded — low diversity"
    elif score > 0.15:
        return "Stressed — anthropogenic pressure"
    else:
        return "Ecological collapse"


def predict(model, clips, idxs, n_mc=30):
    device = next(model.parameters()).device

    clips = clips.to(device)
    idxs = idxs.to(device)

    preds = []
    sp_l = []
    an_l = []
    geo_l = []

    for _ in range(n_mc):
        _enable_dropout(model)

        with torch.no_grad():
            lei, _, scores = model(clips, idxs)

        preds.append(lei.item())
        sp_l.append(scores["species"].mean().item())
        an_l.append(scores["anthro"].mean().item())
        geo_l.append(scores["geo"].mean().item())

    model.eval()

    lei_mean = float(np.mean(preds))
    lei_std = float(np.std(preds))

    return {
        "lei_score": round(lei_mean, 4),
        "uncertainty": round(lei_std, 4),
        "ci_low": round(float(np.percentile(preds, 2.5)), 4),
        "ci_high": round(float(np.percentile(preds, 97.5)), 4),
        "status": _status(lei_mean),
        "components": {
            "species_richness": round(float(np.mean(sp_l)), 4),
            "biophony": round(1 - float(np.mean(an_l)), 4),
            "geophony": round(float(np.mean(geo_l)), 4),
        },
    }