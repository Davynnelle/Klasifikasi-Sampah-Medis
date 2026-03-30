import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import time

# ─── CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="MedWaste Classifier",
    page_icon="🧬",
    layout="wide"
)

MODEL_PATH = "tflite/model.tflite"
LABEL_PATH = "tflite/label.txt"
IMG_SIZE = (224, 224)

# ─── DEBUG INFO (hapus nanti kalau sudah aman) ───────────
st.write("Working dir:", os.getcwd())
st.write("Model exists:", os.path.exists(MODEL_PATH))
st.write("Label exists:", os.path.exists(LABEL_PATH))

# ─── LABELS ──────────────────────────────────────────────
FALLBACK_LABELS = [
    "ampoules_full","ampuoles_broken","blood_soaked_bandages",
    "disinfectant_bottles","episiotomy_scissors","expired_tablets",
    "forceps","general_organic_waste","hemostats","human_organs",
    "iv_bottles","mayo_scissors","scalpels","stitch_removal_scissors",
    "syrup_bottles","tweezers","used_masks","used_medical_gloves",
    "used_medical_paper","used_syringes"
]

@st.cache_data
def load_labels():
    try:
        if os.path.exists(LABEL_PATH):
            with open(LABEL_PATH) as f:
                return [l.strip() for l in f.readlines()]
        return FALLBACK_LABELS
    except Exception as e:
        st.error(f"Error load label: {e}")
        return FALLBACK_LABELS

# ─── LOAD MODEL ──────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"❌ Gagal load model: {e}")
        st.stop()

# ─── PREPROCESS ──────────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ─── PREDICT ─────────────────────────────────────────────
def predict(interpreter, img_array, labels):
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()

        probs = interpreter.get_tensor(output_details[0]["index"])[0]
        top3_idx = np.argsort(probs)[::-1][:3]

        return [(labels[i], float(probs[i]) * 100) for i in top3_idx]

    except Exception as e:
        st.error(f"❌ Error saat prediksi: {e}")
        return []

# ─── UI ──────────────────────────────────────────────────
st.title("🧬 MedWaste Classifier")
st.write("Upload gambar limbah medis untuk diklasifikasikan")

# Load resource
interpreter = load_model()
labels = load_labels()

uploaded = st.file_uploader("Upload gambar", type=["jpg","png","jpeg"])

if uploaded:
    try:
        img = Image.open(uploaded)
    except:
        st.error("❌ File gambar tidak valid")
        st.stop()

    st.image(img, caption="Preview", use_container_width=True)

    if st.button("🔍 Analisis"):
        with st.spinner("Menganalisis..."):
            time.sleep(0.5)

            arr = preprocess(img)
            results = predict(interpreter, arr, labels)

        if results:
            st.subheader("Hasil Prediksi")

            for i, (label, conf) in enumerate(results):
                st.write(f"{i+1}. {label} → {conf:.2f}%")

else:
    st.info("📤 Upload gambar terlebih dahulu")
