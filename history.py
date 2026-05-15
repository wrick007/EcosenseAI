"""
history.py
Save and load analysis history as a CSV file.
"""
from pathlib import Path
from datetime import datetime
import pandas as pd

HISTORY_FILE = Path("history.csv")

COLS = [
    "timestamp", "filename", "location",
    "lei_score", "uncertainty", "status",
    "species_richness", "biophony", "geophony",
    "ACI", "H", "NDSI", "BI",
]


def load() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return pd.DataFrame(columns=COLS)
    try:
        return pd.read_csv(HISTORY_FILE)
    except Exception:
        return pd.DataFrame(columns=COLS)


def save(record: dict):
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new = pd.DataFrame([record])

    if HISTORY_FILE.exists():
        df_old = pd.read_csv(HISTORY_FILE)
        df_out = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_out = df_new

    cols = [c for c in COLS if c in df_out.columns]
    df_out[cols].to_csv(HISTORY_FILE, index=False)


def clear():
    HISTORY_FILE.unlink(missing_ok=True)
