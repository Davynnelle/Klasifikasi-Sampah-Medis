import streamlit as st
import numpy as np
from PIL import Image
import os

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Limbah Medis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS (AMBIL DARI A) ─────────────────────────────────
st.markdown("""
<style>
/* (CSS SAMA PERSIS DARI A — POTONGANMU LANGSUNG TARUH SINI) */
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "tflite", "model.tflite")
LABEL_PATH  = os.path.join(SCRIPT_DIR, "tflite", "label.txt")
IMG_SIZE    = (224, 224)

# ─── Load Model (PAKAI CARA B) ───────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    return interpreter, input_details, output_details

@st.cache_data
def load_labels():
    with open(LABEL_PATH, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]

# ─── Preprocess & Predict (B) ────────────────────────────
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(img, interpreter, input_details, output_details, labels):
    arr = preprocess(img)
    interpreter.set_tensor(input_details[0]["index"], arr)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])[0]
    top3 = np.argsort(output)[::-1][:3]
    return [(labels[i], float(output[i]) * 100) for i in top3]

def fmt_class(name):
    return name.replace("_", " ").title()

# ─── Hero (A) ───────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧬 AI · Medical Waste Detection</div>
    <h1>MedWaste <span>Classifier</span></h1>
    <p>Identifikasi limbah medis otomatis dengan deep learning.</p>
</div>
""", unsafe_allow_html=True)

# ─── Load model ─────────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error("Model atau label tidak ditemukan!")
    st.stop()

with st.spinner("Memuat model..."):
    interpreter, input_details, output_details = load_model()
    labels = load_labels()

# ─── Layout ─────────────────────────────────────────────
col_left, col_right = st.columns([1,1], gap="large")

# ─── LEFT (UPLOAD) ──────────────────────────────────────
with col_left:
    st.markdown('<div class="section-title">Upload Gambar</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload",
        type=["jpg","jpeg","png","webp","bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)
        run_btn = st.button("🔍 Analisis Sekarang", use_container_width=True)
    else:
        run_btn = False

# ─── RIGHT (RESULT) ─────────────────────────────────────
with col_right:
    st.markdown('<div class="section-title">Hasil Klasifikasi</div>', unsafe_allow_html=True)

    if uploaded and run_btn:
        with st.spinner("Menganalisis..."):
            results = predict(img, interpreter, input_details, output_details, labels)

        top_class, top_conf = results[0]

        st.markdown(f"""
        <div class="result-main">
            <div class="result-label">Prediksi Utama</div>
            <div class="result-class">{fmt_class(top_class)}</div>
            <div class="result-conf">
                Confidence
                <span class="conf-pill">{top_conf:.1f}%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Top 3 Prediksi</div>', unsafe_allow_html=True)

        for i, (cls, conf) in enumerate(results):
            st.markdown(f"""
            <div class="top3-item">
                <div class="top3-name">{fmt_class(cls)}</div>
                <div class="top3-bar-bg">
                    <div class="top3-bar" style="width:{conf}%"></div>
                </div>
                <div class="top3-pct">{conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="opacity:0.6;text-align:center;padding:3rem">
            Upload gambar untuk mulai
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem; opacity:0.6;">
MedWaste Classifier · EfficientNetB0 · TF Lite
</div>
""", unsafe_allow_html=True)
