import streamlit as st

st.set_page_config(
    page_title="EcoSense AI",
    page_icon="🌿",
    layout="wide",
)

import os
import tempfile
import time
from pathlib import Path

import audio as aud
import plots
import history as hist
from inference import load_model, predict


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return load_model()


with st.sidebar:
    st.markdown("## 🌿 EcoSense AI")
    st.markdown("Predict ecosystem health from any audio recording.")
    st.markdown("---")

    st.markdown("**How it works**")
    st.markdown(
        "1. Upload a WAV, MP3, FLAC, OGG, or M4A file\n"
        "2. Audio is split into six 5-second clips\n"
        "3. Log-mel spectrograms and acoustic indices are extracted\n"
        "4. The model predicts LEI score with uncertainty"
    )

    st.markdown("---")

    n_mc = st.slider(
        "MC Dropout passes",
        min_value=10,
        max_value=50,
        value=30,
        step=5,
        help="More passes = better uncertainty estimate, but slower inference.",
    )

    st.markdown("---")
    st.markdown("**Score guide**")

    score_guide = [
        ("0.75–1.0", "Pristine", "#2E9955"),
        ("0.55–0.75", "Recovering", "#6ab04c"),
        ("0.35–0.55", "Degraded", "#f9ca24"),
        ("0.15–0.35", "Stressed", "#e17055"),
        ("0.0–0.15", "Collapse", "#d63031"),
    ]

    for rng, label, col in score_guide:
        st.markdown(
            f"<span style='color:{col}; font-weight:600'>■</span> "
            f"**{rng}** — {label}",
            unsafe_allow_html=True,
        )


tab_analyze, tab_history, tab_about = st.tabs(
    ["📊 Analyze", "📅 History", "ℹ️ About"]
)


with tab_analyze:
    st.markdown("### Upload a soundscape recording")
    st.caption("Works on jungle, city, park, ocean, farm, riverbank, or any natural environment.")

    col_upload, col_right = st.columns([1, 1], gap="large")

    uploaded = None
    location = ""

    with col_upload:
        location = st.text_input(
            "Location label",
            placeholder="e.g. Borneo rainforest, Delhi street, Ganga riverbank",
        )

        uploaded = st.file_uploader(
            "Choose audio file",
            type=["wav", "mp3", "flac", "ogg", "m4a"],
            help="Minimum 5 seconds. 30 seconds or more gives the best result.",
        )

        if uploaded:
            st.audio(uploaded)
            st.caption(f"`{uploaded.name}` · {uploaded.size / 1024:.0f} KB")

    if uploaded:
        result = None
        idx = None
        mel_spec = None
        ms = None

        with col_right:
            suffix = Path(uploaded.name).suffix

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded.read())
            tmp.close()

            try:
                with st.spinner("Computing spectrogram and acoustic indices..."):
                    clips, idxs, mel_spec, idx = aud.load_audio(tmp.name)

                st.markdown("**Spectrogram**")
                st.pyplot(plots.spectrogram(mel_spec), use_container_width=True)

                model = get_model()

                with st.spinner(f"Running model with {n_mc} MC Dropout passes..."):
                    t0 = time.time()
                    result = predict(model, clips, idxs, n_mc=n_mc)
                    ms = round((time.time() - t0) * 1000)

            except Exception as e:
                st.error(f"Error: {e}")

            finally:
                if os.path.exists(tmp.name):
                    os.unlink(tmp.name)

        if result is not None:
            st.markdown("---")
            st.markdown("## Results")

            r1, r2, r3 = st.columns([1, 1, 1], gap="large")

            with r1:
                st.markdown("**LEI Score**")
                st.pyplot(
                    plots.gauge(
                        result["lei_score"],
                        result["uncertainty"],
                        result["ci_low"],
                        result["ci_high"],
                        result["status"],
                    ),
                    use_container_width=True,
                )
                st.caption(f"Inference: {ms} ms · {n_mc} MC passes")

            with r2:
                st.markdown("**Ecological components**")
                st.pyplot(
                    plots.components(result["components"]),
                    use_container_width=True,
                )

            with r3:
                st.markdown("**Baseline acoustic indices**")
                st.metric("ACI  (complexity)", f"{idx['ACI']:.0f}")
                st.metric("H    (entropy)", f"{idx['H']:.3f}")
                st.metric("NDSI (bio ratio)", f"{idx['NDSI']:.3f}")
                st.metric("BI   (bioacoustic)", f"{idx['BI']:.1f} dB")
                st.caption("These indices are also passed into the new model.")

            s = result["lei_score"]

            msg = (
                f"**{result['status']}** · "
                f"LEI = {s:.4f} ± {result['uncertainty']:.4f} · "
                f"95% CI [{result['ci_low']:.3f}, {result['ci_high']:.3f}]"
            )

            if s > 0.55:
                st.success(msg)
            elif s > 0.35:
                st.warning(msg)
            else:
                st.error(msg)

            hist.save(
                {
                    "filename": uploaded.name,
                    "location": location or "Unknown",
                    "lei_score": result["lei_score"],
                    "uncertainty": result["uncertainty"],
                    "status": result["status"],
                    "species_richness": result["components"]["species_richness"],
                    "biophony": result["components"]["biophony"],
                    "geophony": result["components"]["geophony"],
                    **idx,
                }
            )


with tab_history:
    st.markdown("### Analysis history")

    df = hist.load()

    if df.empty:
        st.info("No analyses yet. Upload a file on the Analyze tab.")
    else:
        m1, m2, m3, m4 = st.columns(4)

        m1.metric("Total", len(df))
        m2.metric("Mean LEI", f"{df['lei_score'].mean():.3f}")
        m3.metric("Best", f"{df['lei_score'].max():.3f}")
        m4.metric("Worst", f"{df['lei_score'].min():.3f}")

        st.pyplot(plots.history_chart(df), use_container_width=True)

        show_cols = [
            c
            for c in [
                "timestamp",
                "location",
                "filename",
                "lei_score",
                "uncertainty",
                "status",
            ]
            if c in df.columns
        ]

        st.dataframe(
            df[show_cols].sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("Clear history"):
            hist.clear()
            st.rerun()


with tab_about:
    st.markdown("### What is EcoSense AI?")

    st.markdown(
        "EcoSense AI predicts ecosystem health from passive audio using a "
        "deep learning pipeline trained on soundscape data.\n\n"
        "**Stage 1 — AudioViT Encoder**  \n"
        "Learns acoustic representations from log-mel spectrogram clips.\n\n"
        "**Stage 2 — Ecological Head**  \n"
        "Predicts ecological components such as species richness, anthrophony, and geophony.\n\n"
        "**Stage 3 — Temporal LEI Aggregator**  \n"
        "Uses six consecutive 5-second clips to estimate a 30-second ecosystem health score.\n\n"
        "**New app version**  \n"
        "This version also passes normalized acoustic indices into the model."
    )

    st.markdown("---")

    st.markdown("### Score interpretation")

    st.table(
        {
            "Score": [
                "0.75–1.0",
                "0.55–0.75",
                "0.35–0.55",
                "0.15–0.35",
                "0.0–0.15",
            ],
            "Status": [
                "Pristine",
                "Recovering",
                "Degraded",
                "Stressed",
                "Collapse",
            ],
            "Meaning": [
                "High biodiversity, minimal disturbance",
                "Moderate biodiversity, some human influence",
                "Low diversity, habitat degradation visible",
                "Significant anthropogenic pressure",
                "Urban dominance or ecological collapse",
            ],
        }
    )

    st.markdown("---")

    st.markdown("### Acoustic indices used")

    st.table(
        {
            "Index": ["ACI", "H", "NDSI", "BI"],
            "Name": [
                "Acoustic Complexity Index",
                "Acoustic Entropy",
                "Normalized Difference Soundscape Index",
                "Bioacoustic Index",
            ],
            "Role": [
                "Measures sound variation and complexity",
                "Measures entropy/randomness of the soundscape",
                "Compares biological and anthropogenic frequency bands",
                "Measures energy in the biological sound band",
            ],
        }
    )

    st.markdown("---")

    st.markdown("### Tech stack")

    st.code(
        "Model training : PyTorch, AudioViT, Transformer Aggregator\n"
        "Inference      : PyTorch + Monte Carlo Dropout\n"
        "Audio          : torchaudio, librosa\n"
        "App            : Streamlit\n"
        "Storage        : Local CSV history\n"
        "Weights        : models/best.pt",
        language="text",
    )