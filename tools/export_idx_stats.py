"""
Run this in Colab after manifest.csv is created.
It exports the exact min/max stats used to normalize ACI/H/NDSI/BI.
"""

import json
from pathlib import Path
import pandas as pd

BASE = Path('/content/ecosense') if Path('/content').exists() else Path('/kaggle/working/ecosense')
manifest = BASE / 'data' / 'processed' / 'manifest.csv'
out = Path('idx_stats.json')

df = pd.read_csv(manifest)
stats = {
    col: {"min": float(df[col].min()), "max": float(df[col].max())}
    for col in ['ACI', 'H', 'NDSI', 'BI']
}
out.write_text(json.dumps(stats, indent=2))
print(f"Saved {out.resolve()}")
print(json.dumps(stats, indent=2))
