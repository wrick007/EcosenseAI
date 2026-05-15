"""
plots.py
All visualization functions.
Each returns a matplotlib Figure for st.pyplot().
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

G   = "#1A6B3C"
RED = "#d63031"


def _score_color(s: float) -> str:
    if s > 0.75: return "#2E9955"
    if s > 0.55: return "#6ab04c"
    if s > 0.35: return "#f9ca24"
    if s > 0.15: return "#e17055"
    return RED


def gauge(score, uncertainty, ci_low, ci_high, status) -> plt.Figure:
    """Semicircle gauge with needle."""
    fig, ax = plt.subplots(figsize=(4.8, 3.0), facecolor="none")
    ax.axis("off"); ax.set_aspect("equal")

    # Colour bands
    for (s, e, c) in [
        (np.pi,        np.pi * 0.80, "#d63031"),
        (np.pi * 0.80, np.pi * 0.60, "#e17055"),
        (np.pi * 0.60, np.pi * 0.40, "#f9ca24"),
        (np.pi * 0.40, np.pi * 0.20, "#6ab04c"),
        (np.pi * 0.20, 0.0,          "#2E9955"),
    ]:
        t = np.linspace(s, e, 60)
        ax.plot(np.cos(t), np.sin(t), color=c, lw=16, alpha=0.85, solid_capstyle="butt")

    # Uncertainty arc
    t_ci = np.linspace(np.pi * (1 - ci_high), np.pi * (1 - ci_low), 40)
    ax.fill_between(np.cos(t_ci), np.sin(t_ci) * 0.76,
                    np.sin(t_ci) * 1.0, color="white", alpha=0.30)

    # Needle
    col   = _score_color(score)
    angle = np.pi * (1 - score)
    ax.annotate("",
        xy=(0.70 * np.cos(angle), 0.70 * np.sin(angle)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", lw=2.5, color=col, mutation_scale=12))
    ax.plot(0, 0, "o", color=col, ms=7, zorder=6)

    # Labels
    ax.text(0, -0.18, f"{score:.3f}", ha="center", fontsize=26,
            fontweight="bold", color=col)
    ax.text(0, -0.36, f"±{uncertainty:.3f}   [{ci_low:.2f} – {ci_high:.2f}]",
            ha="center", fontsize=9, color="#777")
    ax.text(0, 0.22, status, ha="center", fontsize=8.5,
            color=col, style="italic")

    for v, lbl in [(0.0, "0"), (0.5, "0.5"), (1.0, "1.0")]:
        a = np.pi * (1 - v)
        ax.text(1.14 * np.cos(a), 1.14 * np.sin(a), lbl,
                ha="center", va="center", fontsize=7.5, color="#aaa")

    ax.set_xlim(-1.25, 1.25); ax.set_ylim(-0.55, 1.15)
    fig.tight_layout(pad=0.2)
    return fig


def spectrogram(mel: np.ndarray) -> plt.Figure:
    """Log-mel spectrogram heatmap."""
    fig, ax = plt.subplots(figsize=(9, 2.2), facecolor="none")
    im = ax.imshow(mel, aspect="auto", origin="lower",
                   cmap="magma", interpolation="nearest", vmin=-3, vmax=3)
    ax.set_xlabel("Time frames", fontsize=8)
    ax.set_ylabel("Mel bins",    fontsize=8)
    ax.set_title("Log-Mel Spectrogram", fontsize=9, pad=4)
    plt.colorbar(im, ax=ax, label="norm. dB", fraction=0.02, pad=0.01)
    fig.tight_layout(pad=0.4)
    return fig


def components(comp: dict) -> plt.Figure:
    """Horizontal bars for species richness, biophony, geophony."""
    labels = ["Species\nRichness", "Biophony", "Geophony"]
    values = [comp["species_richness"], comp["biophony"], comp["geophony"]]
    colors = [G, "#6ab04c", "#74b9ff"]

    fig, ax = plt.subplots(figsize=(4.2, 2.5), facecolor="none")
    bars = ax.barh(labels, values, color=colors, edgecolor="white", height=0.45)
    ax.set_xlim(0, 1.18)
    ax.set_xlabel("Score [0–1]", fontsize=8)
    ax.axvline(0.5, color="#ddd", lw=0.8, ls="--")
    for bar, v in zip(bars, values):
        ax.text(v + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout(pad=0.4)
    return fig


def history_chart(df: pd.DataFrame) -> plt.Figure:
    """LEI over time with uncertainty band."""
    fig, ax = plt.subplots(figsize=(9, 3.0), facecolor="none")
    x   = range(len(df))
    lei = df["lei_score"].values
    unc = df["uncertainty"].values if "uncertainty" in df.columns else np.zeros(len(df))

    ax.fill_between(x, lei - unc, lei + unc, alpha=0.2, color=G, label="±1σ")
    ax.plot(x, lei, color=G, lw=2, marker="o", ms=5, label="LEI score")
    ax.axhline(0.5, color="#ccc", ls="--", lw=0.8)

    # Background zones
    for lo, hi, col in [(0.0,0.15,RED),(0.15,0.35,"#e17055"),
                         (0.35,0.55,"#f9ca24"),(0.55,0.75,"#6ab04c"),
                         (0.75,1.0, "#2E9955")]:
        ax.axhspan(lo, hi, alpha=0.05, color=col)

    ax.set_ylim(0, 1); ax.set_xlim(0, max(len(df) - 1, 1))
    ax.set_ylabel("LEI Score", fontsize=8)
    ax.set_title("LEI Score Over Time", fontsize=10)
    ax.legend(fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    if "location" in df.columns and len(df) <= 12:
        ax.set_xticks(list(x))
        ax.set_xticklabels(df["location"].values,
                           rotation=25, ha="right", fontsize=7)
    fig.tight_layout(pad=0.4)
    return fig
