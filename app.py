import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
import time
import traceback
import tflite_runtime.interpreter as tflite

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedWaste AI · Klasifikasi Limbah Medis",
    page_icon="🧬",
    layout="wide"
)

# ─── Constants ───────────────────────────────────────────────
MODEL_PATH = "tflite/model.tflite"
LABEL_PATH = "tflite/label.txt"
IMG_SIZE = (224, 224)

FALLBACK_LABELS = [
    "ampoules_full", "ampuoles_broken", "blood_soaked_bandages",
    "disinfectant_bottles", "episiotomy_scissors", "expired_tablets",
    "forceps", "general_organic_waste", "hemostats", "human_organs",
    "iv_bottles", "mayo_scissors", "scalpels", "stitch_removal_scissors",
    "syrup_bottles", "tweezers", "used_masks", "used_medical_gloves",
    "used_medical_paper", "used_syringes"
]

# ─── Load Model (SAFE) ───────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            return [line.strip() for line in f.readlines() if line.strip()]
    return FALLBACK_LABELS

# ─── Preprocess ──────────────────────────────────────────────
def preprocess(img: Image.Image):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)

# ─── Predict ─────────────────────────────────────────────────
def predict(interpreter, img_array, labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]['index'])[0]
    top3 = np.argsort(probs)[::-1][:3]

    return [(labels[i], float(probs[i]) * 100) for i in top3]

# ─── Load resources with DEBUG ───────────────────────────────
try:
    interpreter = load_model()
    labels = load_labels()
    model_ok = True
except Exception as e:
    model_ok = False
    st.error("❌ Gagal load model")
    st.text(traceback.format_exc())

# ─── UI ─────────────────────────────────────────────────────
st.title("🧬 MedWaste Classifier")

uploaded = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img)

    if st.button("Analisis"):
        if not model_ok:
            st.warning("Model tidak tersedia.")
        else:
            with st.spinner("Processing..."):
                arr = preprocess(img)
                results = predict(interpreter, arr, labels)

            st.success("Hasil:")
            for cls, conf in results:
                st.write(f"{cls} — {conf:.2f}%")
