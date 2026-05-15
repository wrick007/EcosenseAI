# EcoSense AI Streamlit App

This version matches the newer Colab notebook architecture where inference is:

```python
model(clips, idxs)
```

## Required files

```text
app.py
audio.py
inference.py
model.py
plots.py
history.py
idx_stats.json
models/best.pt
requirements.txt
```

## From Colab

1. Copy your checkpoint:

```text
/content/ecosense/checkpoints/lei/best.pt
```

or if you intentionally use the retrained file:

```text
/content/ecosense/checkpoints/lei/best_real.pt
```

Rename the chosen file to:

```text
models/best.pt
```

2. Export index stats in Colab:

```python
!python tools/export_idx_stats.py
```

Download `idx_stats.json` and put it beside `app.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud

Upload weights to Hugging Face and set secrets:

```toml
HF_REPO = "your-username/ecosense-ai"
HF_FILENAME = "best.pt"
```

Keep `idx_stats.json` in the GitHub repo.
