"""
app.py - tampilan web klasifikasi motif batik (Streamlit).
Jalankan dari root repo:  streamlit run app/app.py
"""
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image

from predict import MODEL_NAMES, is_real, predict

SAMPLES_DIR = Path(__file__).resolve().parent / "samples"
IMG_EXT = {".jpg", ".jpeg", ".png"}
TOP_N = 5  # jumlah kelas teratas yang ditampilkan di grafik
LOW_CONFIDENCE = 0.5  # di bawah ini, tampilkan peringatan

st.set_page_config(page_title="Klasifikasi Motif Batik", page_icon="🧵", layout="wide")

st.title("🧵 Klasifikasi Motif Batik")
st.caption("Bandingkan GLCM + SVM, MobileNetV2, dan ResNet50")

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Pengaturan")
    model_name = st.selectbox("Pilih model", MODEL_NAMES)
    if is_real(model_name):
        st.success("Model asli dimuat")
    else:
        st.warning("Model belum tersedia, hasil masih ACAK (demo)")
    compare_all = st.checkbox("Bandingkan semua model")

# -------------------------------------------------------------- input gambar
col_in, col_out = st.columns(2)

image = None
with col_in:
    st.subheader("1. Pilih gambar")
    uploaded = st.file_uploader("Upload gambar batik", type=["jpg", "jpeg", "png"])

    samples = (
        sorted(p for p in SAMPLES_DIR.iterdir() if p.suffix.lower() in IMG_EXT)
        if SAMPLES_DIR.exists()
        else []
    )
    sample_choice = None
    if samples:
        sample_choice = st.selectbox(
            "atau pilih gambar contoh",
            ["(tidak dipilih)"] + [p.name for p in samples],
        )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
    elif sample_choice and sample_choice != "(tidak dipilih)":
        image = Image.open(SAMPLES_DIR / sample_choice).convert("RGB")

    if image is not None:
        st.image(image, caption="Gambar input", width="stretch")

    run = st.button("Klasifikasi", type="primary", disabled=image is None)

# ------------------------------------------------------------------- hasil
with col_out:
    st.subheader("2. Hasil")
    if image is None:
        st.info("Upload gambar atau pilih gambar contoh, lalu klik Klasifikasi.")
    elif run:
        with st.spinner("Memproses..."):
            result = predict(image, model_name)

        st.metric("Motif terdeteksi", result["label"], f"{result['confidence'] * 100:.2f}%")
        st.caption(
            f"Model: {model_name} | Waktu prediksi: {result['elapsed'] * 1000:.1f} ms"
            + (" | MODE DEMO (hasil acak)" if result["mode"] == "palsu" else "")
        )

        if result["confidence"] < LOW_CONFIDENCE:
            st.warning(
                "Keyakinan model rendah. Gambar mungkin bukan motif batik "
                "atau motifnya tidak termasuk dalam kelas dataset."
            )

        st.markdown(f"**{TOP_N} kelas teratas**")
        df = pd.DataFrame(
            {
                "Motif": list(result["probabilities"].keys()),
                "Probabilitas (%)": [p * 100 for p in result["probabilities"].values()],
            }
        )
        chart = (
            alt.Chart(df.head(TOP_N))
            .mark_bar()
            .encode(
                x=alt.X("Probabilitas (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Motif:N", sort="-x", title=None),
                tooltip=["Motif", alt.Tooltip("Probabilitas (%):Q", format=".2f")],
            )
            .properties(height=45 * min(TOP_N, len(df)) + 20)
        )
        st.altair_chart(chart, width="stretch")

        with st.expander("Lihat probabilitas semua kelas"):
            st.dataframe(
                df.style.format({"Probabilitas (%)": "{:.2f}"}),
                width="stretch",
                hide_index=True,
            )

        if compare_all:
            st.markdown("**Perbandingan semua model**")
            rows = []
            for name in MODEL_NAMES:
                r = predict(image, name)
                rows.append(
                    {
                        "Model": name,
                        "Prediksi": r["label"],
                        "Confidence (%)": round(r["confidence"] * 100, 2),
                        "Waktu (ms)": round(r["elapsed"] * 1000, 1),
                        "Mode": r["mode"],
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
