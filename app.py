import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import os
import time

# ─── Page Config ─────────────────────────────────────────
st.set_page_config(
    page_title="Klasifikasi Limbah Medis",
    page_icon="🧬",
    layout="wide",
)

# ─── Paths ───────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "tflite", "model.tflite")
LABEL_PATH = os.path.join(SCRIPT_DIR, "tflite", "label.txt")
IMG_SIZE = (224, 224)

# ─── Labels fallback ─────────────────────────────────────
FALLBACK_LABELS = [
    "ampoules_full","ampuoles_broken","blood_soaked_bandages",
    "disinfectant_bottles","episiotomy_scissors","expired_tablets",
    "forceps","general_organic_waste","hemostats","human_organs",
    "iv_bottles","mayo_scissors","scalpels","stitch_removal_scissors",
    "syrup_bottles","tweezers","used_masks","used_medical_gloves",
    "used_medical_paper","used_syringes"
]

# ─── Load Model ──────────────────────────────────────────
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model tidak ditemukan di folder /tflite/")
        st.stop()

    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_data
def load_labels():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH) as f:
            return [l.strip() for l in f.readlines()]
    return FALLBACK_LABELS

# ─── Preprocess ──────────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0  # penting!
    return np.expand_dims(arr, axis=0)

# ─── Predict ─────────────────────────────────────────────
def predict(interpreter, img_array, labels):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    top3_idx = np.argsort(probs)[::-1][:3]

    return [(labels[i], float(probs[i]) * 100) for i in top3_idx]

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
        st.error("File tidak valid")
        st.stop()

    st.image(img, caption="Preview", use_container_width=True)

    if st.button("Analisis"):
        with st.spinner("Processing..."):
            time.sleep(0.5)

            arr = preprocess(img)
            results = predict(interpreter, arr, labels)

        st.subheader("Hasil Prediksi")

        for i, (label, conf) in enumerate(results):
            st.write(f"{i+1}. {label} → {conf:.2f}%")

else:
    st.info("Upload gambar dulu ya 👆")
