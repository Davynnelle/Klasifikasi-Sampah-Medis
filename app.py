import streamlit as st
import numpy as np
from PIL import Image
import os
import time

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Limbah Medis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── CSS (FULL DARI A - JANGAN DIUBAH) ───────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg:#080c12; --surface:#0e1520; --card:#111927;
    --border:#1e2d42; --accent:#00d4aa; --accent2:#0090ff;
    --danger:#ff4d6d; --warning:#ffb830;
    --text:#e8edf5; --muted:#6b7fa3; --radius:16px;
}
* { font-family:'DM Sans',sans-serif; }
html,body,[class*="css"] { background-color:var(--bg)!important;color:var(--text)!important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding:2rem 3rem!important; max-width:1200px; }

/* HERO */
.hero {
    background:linear-gradient(135deg,#0a1628,#0d1f35,#091520);
    border:1px solid var(--border);
    border-radius:24px;
    padding:3rem 3.5rem;
    margin-bottom:2rem;
}

/* SECTION */
.section-title {
    font-family:'Syne',sans-serif;
    font-size:1rem;
    font-weight:700;
    text-transform:uppercase;
    margin-bottom:1rem;
}

/* RESULT */
.result-main {
    background:linear-gradient(135deg,rgba(0,212,170,0.08),rgba(0,144,255,0.05));
    border:1px solid rgba(0,212,170,0.25);
    border-radius:16px;
    padding:2rem;
}
.result-class {
    font-family:'Syne',sans-serif;
    font-size:2rem;
    font-weight:800;
}
.conf-pill {
    background:rgba(0,212,170,0.15);
    padding:2px 10px;
    border-radius:100px;
}

/* TOP3 */
.top3-item {
    background:var(--card);
    border:1px solid var(--border);
    border-radius:10px;
    padding:1rem;
    margin-bottom:0.6rem;
}
.top3-bar-bg {
    background:var(--border);
    height:4px;
    margin-top:6px;
}
.top3-bar {
    height:4px;
    background:linear-gradient(90deg,var(--accent),var(--accent2));
}
.top3-pct {
    float:right;
}

/* BUTTON */
.stButton>button {
    background:linear-gradient(135deg,var(--accent),var(--accent2));
    color:#000;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ─── Constants ───────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "tflite", "model.tflite")
LABEL_PATH  = os.path.join(SCRIPT_DIR, "tflite", "label.txt")
IMG_SIZE    = (224, 224)

# ─── Load Model (B VERSION) ──────────────────────────────
@st.cache_resource
def load_model():
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return (
        interpreter,
        interpreter.get_input_details(),
        interpreter.get_output_details()
    )

@st.cache_data
def load_labels():
    with open(LABEL_PATH) as f:
        return [l.strip() for l in f.readlines()]

# ─── Inference ───────────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(img, interpreter, inp, out, labels):
    arr = preprocess(img)
    interpreter.set_tensor(inp[0]["index"], arr)
    interpreter.invoke()
    probs = interpreter.get_tensor(out[0]["index"])[0]
    top3 = np.argsort(probs)[::-1][:3]
    return [(labels[i], float(probs[i]) * 100) for i in top3]

def fmt(x):
    return x.replace("_", " ").title()

# ─── HERO ────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>MedWaste Classifier</h1>
    <p>AI untuk klasifikasi limbah medis berbasis EfficientNetB0</p>
</div>
""", unsafe_allow_html=True)

# ─── Load Model ──────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("Model tidak ditemukan")
    st.stop()

interpreter, inp, out = load_model()
labels = load_labels()

# ─── Layout ─────────────────────────────────────────────
col1, col2 = st.columns(2)

# LEFT
with col1:
    st.markdown('<div class="section-title">Upload Gambar</div>', unsafe_allow_html=True)

    file = st.file_uploader("", type=["jpg","png","jpeg"])
    if file:
        img = Image.open(file)
        st.image(img, use_container_width=True)
        run = st.button("🔍 Analisis")
    else:
        run = False

# RIGHT
with col2:
    st.markdown('<div class="section-title">Hasil</div>', unsafe_allow_html=True)

    if file and run:
        with st.spinner("Analyzing..."):
            time.sleep(0.3)
            res = predict(img, interpreter, inp, out, labels)

        top, conf = res[0]

        st.markdown(f"""
        <div class="result-main">
            <div class="result-class">{fmt(top)}</div>
            <div>Confidence <span class="conf-pill">{conf:.1f}%</span></div>
        </div>
        """, unsafe_allow_html=True)

        for cls, c in res:
            st.markdown(f"""
            <div class="top3-item">
                {fmt(cls)}
                <div class="top3-bar-bg">
                    <div class="top3-bar" style="width:{c}%"></div>
                </div>
                <div class="top3-pct">{c:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

# ─── Footer ─────────────────────────────────────────────
st.markdown("<center><sub>EfficientNetB0 · TF Lite</sub></center>", unsafe_allow_html=True)
