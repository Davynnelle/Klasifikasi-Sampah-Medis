import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Limbah Medis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg:        #080c12;
    --surface:   #0e1520;
    --card:      #111927;
    --border:    #1e2d42;
    --accent:    #00d4aa;
    --accent2:   #0090ff;
    --danger:    #ff4d6d;
    --warning:   #ffb830;
    --text:      #e8edf5;
    --muted:     #6b7fa3;
    --radius:    16px;
}

* { font-family: 'DM Sans', sans-serif; }

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem !important; max-width: 1200px; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0a1628 0%, #0d1f35 50%, #091520 100%);
    border: 1px solid var(--border);
    border-radius: 24px;
    padding: 3rem 3.5rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,212,170,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: -40px;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,144,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(0,212,170,0.12);
    border: 1px solid rgba(0,212,170,0.3);
    color: var(--accent);
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.hero h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    color: var(--text) !important;
    margin: 0 0 0.75rem 0 !important;
}
.hero h1 span {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero p {
    color: var(--muted) !important;
    font-size: 1.05rem !important;
    font-weight: 300 !important;
    max-width: 560px !important;
    line-height: 1.65 !important;
    margin: 0 !important;
}

/* ── Stats row ── */
.stats-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    flex: 1;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 4px;
}
.stat-label {
    font-size: 0.72rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.stat-value {
    font-family: 'Syne', sans-serif;
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
}
.stat-sub {
    font-size: 0.75rem;
    color: var(--muted);
}

/* ── Upload zone ── */
.upload-section {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 18px;
    background: linear-gradient(to bottom, var(--accent), var(--accent2));
    border-radius: 2px;
}

/* ── Result card ── */
.result-main {
    background: linear-gradient(135deg, rgba(0,212,170,0.08), rgba(0,144,255,0.05));
    border: 1px solid rgba(0,212,170,0.25);
    border-radius: var(--radius);
    padding: 2rem;
    margin-bottom: 1.5rem;
}
.result-label {
    font-size: 0.72rem;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 600;
    margin-bottom: 0.4rem;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--text);
    margin-bottom: 0.3rem;
    line-height: 1.2;
}
.result-conf {
    font-size: 0.95rem;
    color: var(--muted);
}
.conf-pill {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    color: var(--accent);
    border: 1px solid rgba(0,212,170,0.3);
    padding: 2px 12px;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-left: 8px;
}

/* ── Top3 bars ── */
.top3-item {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    position: relative;
    overflow: hidden;
}
.top3-rank {
    font-family: 'Syne', sans-serif;
    font-size: 0.65rem;
    font-weight: 700;
    color: var(--muted);
    letter-spacing: 0.12em;
    text-transform: uppercase;
}
.top3-name {
    font-size: 0.95rem;
    font-weight: 500;
    color: var(--text);
    margin: 2px 0;
}
.top3-bar-bg {
    background: var(--border);
    border-radius: 4px;
    height: 4px;
    margin-top: 6px;
    overflow: hidden;
}
.top3-bar {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.top3-pct {
    position: absolute;
    right: 1.1rem;
    top: 50%;
    transform: translateY(-50%);
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text);
}

/* ── Info box ── */
.info-box {
    background: rgba(255,184,48,0.07);
    border: 1px solid rgba(255,184,48,0.2);
    border-radius: var(--radius);
    padding: 1rem 1.4rem;
    margin-top: 1rem;
}
.info-box p {
    color: #c9933a !important;
    font-size: 0.85rem !important;
    margin: 0 !important;
}

/* ── Category pill ── */
.cat-danger { color: var(--danger); background: rgba(255,77,109,0.1); border: 1px solid rgba(255,77,109,0.25); }
.cat-caution { color: var(--warning); background: rgba(255,184,48,0.1); border: 1px solid rgba(255,184,48,0.25); }
.cat-safe { color: var(--accent); background: rgba(0,212,170,0.1); border: 1px solid rgba(0,212,170,0.25); }
.category-tag {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 100px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 6px;
}

/* ── Streamlit overrides ── */
.stFileUploader > div {
    background: var(--surface) !important;
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius) !important;
    transition: border-color 0.2s;
}
.stFileUploader > div:hover {
    border-color: var(--accent) !important;
}
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: #000 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s !important;
    letter-spacing: 0.04em !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(0,212,170,0.3) !important;
}
.stSpinner > div { border-top-color: var(--accent) !important; }

div[data-testid="stImage"] img {
    border-radius: var(--radius) !important;
    border: 1px solid var(--border) !important;
}

/* Divider */
hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ─── Constants ──────────────────────────────────────────────────────────────
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(SCRIPT_DIR, "tflite", "model.tflite")
LABEL_PATH  = os.path.join(SCRIPT_DIR, "tflite", "label.txt")
IMG_SIZE    = (224, 224)

FALLBACK_LABELS = [
    "ampoules_full", "ampuoles_broken", "blood_soaked_bandages",
    "disinfectant_bottles", "episiotomy_scissors", "expired_tablets",
    "forceps", "general_organic_waste", "hemostats", "human_organs",
    "iv_bottles", "mayo_scissors", "scalpels", "stitch_removal_scissors",
    "syrup_bottles", "tweezers", "used_masks", "used_medical_gloves",
    "used_medical_paper", "used_syringes"
]

RISK_MAP = {
    "scalpels": ("danger", "⚠️ Benda Tajam Infeksius"),
    "episiotomy_scissors": ("danger", "⚠️ Benda Tajam"),
    "mayo_scissors": ("danger", "⚠️ Benda Tajam"),
    "stitch_removal_scissors": ("danger", "⚠️ Benda Tajam"),
    "hemostats": ("danger", "⚠️ Benda Tajam"),
    "used_syringes": ("danger", "⚠️ Benda Tajam Infeksius"),
    "blood_soaked_bandages": ("danger", "🩸 Limbah Infeksius"),
    "human_organs": ("danger", "🧬 Limbah Patologis"),
    "used_medical_gloves": ("caution", "🧤 Limbah Infeksius"),
    "used_masks": ("caution", "😷 Limbah Infeksius"),
    "forceps": ("caution", "🔧 Peralatan Medis"),
    "tweezers": ("caution", "🔧 Peralatan Medis"),
    "expired_tablets": ("caution", "💊 Limbah Farmasi"),
    "ampoules_full": ("caution", "💉 Limbah Farmasi"),
    "ampuoles_broken": ("caution", "💉 Benda Tajam Farmasi"),
    "iv_bottles": ("caution", "🧴 Limbah Farmasi"),
    "syrup_bottles": ("caution", "🧴 Limbah Farmasi"),
    "disinfectant_bottles": ("safe", "🫧 Limbah Kimia"),
    "general_organic_waste": ("safe", "🌿 Limbah Organik"),
    "used_medical_paper": ("safe", "📄 Limbah Umum"),
}

# ─── Load resources ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return FALLBACK_LABELS

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(interpreter, img_array, labels):
    inp = interpreter.get_input_details()
    out = interpreter.get_output_details()
    interpreter.set_tensor(inp[0]["index"], img_array)
    interpreter.invoke()
    probs   = interpreter.get_tensor(out[0]["index"])[0]
    top3    = np.argsort(probs)[::-1][:3]
    return [(labels[i], float(probs[i]) * 100) for i in top3]

def fmt_class(name: str) -> str:
    return name.replace("_", " ").title()

# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🧬 AI · Medical Waste Detection</div>
    <h1>MedWaste <span>Classifier</span></h1>
    <p>Identifikasi jenis limbah biomedis secara otomatis menggunakan deep learning berbasis EfficientNetB0 — akurasi 98% pada 20 kategori limbah medis.</p>
</div>
""", unsafe_allow_html=True)

# ─── Stats ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-label">Akurasi Model</div>
        <div class="stat-value">98%</div>
        <div class="stat-sub">Test set · EfficientNetB0</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Jumlah Kelas</div>
        <div class="stat-value">20</div>
        <div class="stat-sub">Kategori limbah biomedis</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Dataset</div>
        <div class="stat-value">12.8K</div>
        <div class="stat-sub">Gambar training · Kaggle</div>
    </div>
    <div class="stat-card">
        <div class="stat-label">Format Model</div>
        <div class="stat-value">TFLite</div>
        <div class="stat-sub">Optimized · Mobile-ready</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─── Load model ──────────────────────────────────────────────────────────────
try:
    interpreter = load_model()
    labels      = load_labels()
    model_ok    = True
except Exception as e:
    model_ok = False
    st.error(f"Gagal memuat model: {e}")

# ─── Main layout ─────────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown('<div class="section-title">Upload Gambar</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Pilih gambar limbah medis",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        label_visibility="collapsed"
    )

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("🔍  Analisis Sekarang", use_container_width=True)
    else:
        st.markdown("""
        <div class="info-box">
            <p>📎 Upload foto limbah medis dalam format JPG, PNG, atau WEBP untuk memulai klasifikasi.</p>
        </div>
        """, unsafe_allow_html=True)
        run_btn = False

with col_right:
    st.markdown('<div class="section-title">Hasil Klasifikasi</div>', unsafe_allow_html=True)

    if uploaded and run_btn and model_ok:
        with st.spinner("Menganalisis gambar..."):
            time.sleep(0.4)
            arr     = preprocess(img)
            results = predict(interpreter, arr, labels)

        top_class, top_conf = results[0]
        risk_level, risk_label = RISK_MAP.get(top_class, ("safe", "📦 Limbah Umum"))
        cat_cls = f"cat-{risk_level}"

        st.markdown(f"""
        <div class="result-main">
            <div class="result-label">Prediksi Utama</div>
            <div class="result-class">{fmt_class(top_class)}</div>
            <div class="result-conf">
                Confidence
                <span class="conf-pill">{top_conf:.1f}%</span>
            </div>
            <span class="category-tag {cat_cls}">{risk_label}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title" style="margin-top:1.5rem">Top 3 Prediksi</div>', unsafe_allow_html=True)

        colors = ["var(--accent)", "var(--accent2)", "var(--muted)"]
        ranks  = ["1st", "2nd", "3rd"]

        for idx, (cls_name, conf) in enumerate(results):
            bar_w = max(conf, 3)
            st.markdown(f"""
            <div class="top3-item">
                <div class="top3-rank">{ranks[idx]}</div>
                <div class="top3-name">{fmt_class(cls_name)}</div>
                <div class="top3-bar-bg">
                    <div class="top3-bar" style="width:{bar_w}%; background: {colors[idx]};"></div>
                </div>
                <div class="top3-pct" style="color:{colors[idx]}">{conf:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        if top_conf < 60:
            st.markdown("""
            <div class="info-box">
                <p>⚠️ <strong>Confidence rendah.</strong> Gambar mungkin tidak termasuk dalam 20 kelas yang dikenal, atau kualitas gambar kurang optimal. Coba dengan gambar yang lebih jelas.</p>
            </div>
            """, unsafe_allow_html=True)

    elif not uploaded:
        st.markdown("""
        <div style="background:var(--card); border:1px dashed var(--border); border-radius:var(--radius); padding:3rem 2rem; text-align:center; color:var(--muted);">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">🔬</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:var(--text); margin-bottom:0.4rem;">Belum ada gambar</div>
            <div style="font-size:0.85rem;">Upload gambar di sebelah kiri untuk memulai analisis</div>
        </div>
        """, unsafe_allow_html=True)
    elif not run_btn:
        st.markdown("""
        <div style="background:var(--card); border:1px dashed var(--border); border-radius:var(--radius); padding:3rem 2rem; text-align:center; color:var(--muted);">
            <div style="font-size:2.5rem; margin-bottom:0.75rem;">▶️</div>
            <div style="font-family:'Syne',sans-serif; font-size:1rem; font-weight:600; color:var(--text); margin-bottom:0.4rem;">Siap dianalisis</div>
            <div style="font-size:0.85rem;">Klik tombol "Analisis Sekarang" untuk memulai</div>
        </div>
        """, unsafe_allow_html=True)

# ─── 20 Classes Reference ────────────────────────────────────────────────────
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-title">20 Kelas yang Dapat Dikenali</div>', unsafe_allow_html=True)

cols = st.columns(4)
for i, label in enumerate(FALLBACK_LABELS):
    risk_level, risk_label = RISK_MAP.get(label, ("safe", "📦 Limbah Umum"))
    cat_cls = f"cat-{risk_level}"
    with cols[i % 4]:
        st.markdown(f"""
        <div style="background:var(--card); border:1px solid var(--border); border-radius:10px; padding:0.75rem 1rem; margin-bottom:0.6rem;">
            <div style="font-size:0.8rem; font-weight:500; color:var(--text); margin-bottom:4px;">{fmt_class(label)}</div>
            <span class="category-tag {cat_cls}" style="font-size:0.65rem; padding:1px 8px;">{risk_label}</span>
        </div>
        """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:2rem 0 1rem; color:var(--muted); font-size:0.8rem;">
    MedWaste Classifier · Dibuat oleh <strong style="color:var(--text)">Dhea Yuza Fadiya</strong>
    · Powered by EfficientNetB0 + TFLite
    <br><span style="font-size:0.72rem; opacity:0.6;">© 2025 · Belajar Fundamental Deep Learning · Dicoding Indonesia</span>
</div>
""", unsafe_allow_html=True)
