import streamlit as st
import numpy as np
from PIL import Image
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Biomedical Waste Classifier",
    page_icon="🧬",
    layout="centered",
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH = "submission/tflite/model.tflite"
LABEL_PATH = "submission/tflite/label.txt"
IMG_SIZE   = (224, 224)

# ── Load model & labels (cached) ─────────────────────────────────────────────
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

# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

def predict(image: Image.Image, interpreter, input_details, output_details, class_names):
    img_array = preprocess(image)
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    output   = interpreter.get_tensor(output_details[0]["index"])[0]
    top3_idx = np.argsort(output)[::-1][:3]
    return [(class_names[i], float(output[i]) * 100) for i in top3_idx]

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🧬 Biomedical Waste Classifier")
st.markdown(
    "Upload gambar limbah biomedis, model akan mengklasifikasikan ke **20 kategori** "
    "menggunakan **EfficientNetB0** (Transfer Learning)."
)
st.divider()

# Check files exist
if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_PATH):
    st.error(
        f"❌ Model atau label tidak ditemukan.\n\n"
        f"Pastikan file berikut ada:\n"
        f"- `{MODEL_PATH}`\n"
        f"- `{LABEL_PATH}`"
    )
    st.stop()

# Load
with st.spinner("Memuat model..."):
    interpreter, input_details, output_details = load_model()
    class_names = load_labels()

st.success(f"✅ Model siap — {len(class_names)} kelas terdeteksi")
st.divider()

# Upload
uploaded_file = st.file_uploader(
    "📁 Upload gambar limbah biomedis",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.subheader("🖼️ Gambar Input")
        st.image(image, use_container_width=True)
        st.caption(f"Ukuran: {image.size[0]} × {image.size[1]} px")

    with col2:
        st.subheader("🔍 Hasil Prediksi")
        with st.spinner("Menganalisis gambar..."):
            results = predict(image, interpreter, input_details, output_details, class_names)

        top_label, top_conf = results[0]

        # Badge warna berdasarkan confidence
        if top_conf >= 80:
            badge_color = "green"
        elif top_conf >= 50:
            badge_color = "orange"
        else:
            badge_color = "red"

        st.markdown(
            f"**Prediksi utama:** "
            f":{badge_color}[{top_label.replace('_', ' ').title()}]"
        )
        st.metric(label="Confidence", value=f"{top_conf:.1f}%")

        st.markdown("**Top-3 Prediksi:**")
        colors = ["🥇", "🥈", "🥉"]
        for medal, (label, conf) in zip(colors, results):
            clean_label = label.replace("_", " ").title()
            st.write(f"{medal} **{clean_label}**")
            st.progress(conf / 100, text=f"{conf:.1f}%")

    st.divider()
    with st.expander("📋 Semua kelas yang didukung"):
        cols = st.columns(4)
        for i, name in enumerate(class_names):
            cols[i % 4].write(f"• {name.replace('_', ' ').title()}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    "<br><center><sub>Dibuat oleh Dhea Yuza Fadiya · EfficientNetB0 · TF-Lite</sub></center>",
    unsafe_allow_html=True,
)
