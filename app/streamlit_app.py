"""Streamlit demo. Run: streamlit run app/streamlit_app.py"""
import io
from pathlib import Path

import numpy as np
import streamlit as st

from soilclf.config import load_config
from soilclf.images import load_rgb

ROOT = Path(__file__).resolve().parents[1]
cfg = load_config(ROOT / "configs" / "default.yaml")
CLASSES, SIZE = cfg["data"]["classes"], cfg["data"]["image_size"]
THRESHOLD = cfg["evaluate"]["uncertain_threshold"]
MODEL_DIR = ROOT / cfg["paths"]["artifacts"] / "cnn"


@st.cache_resource
def get_predictor():
    """Prefer the fp16 TFLite model (smallest without slowing CPU inference); fall back to Keras."""
    tflite = MODEL_DIR / "model_fp16.tflite"
    if tflite.exists():
        from soilclf.export import TFLiteRunner

        runner = TFLiteRunner(tflite, SIZE)
        return "TFLite fp16", lambda x: runner(x.astype(np.float32))
    from soilclf.cnn import load_model, predict_proba

    model = load_model(MODEL_DIR / "model.keras")
    return "Keras", lambda x: predict_proba(model, x[None])[0]


st.title("Soil Type Classifier")
st.caption("MobileNetV2 transfer learning · 3 classes: " + ", ".join(CLASSES))

if not (MODEL_DIR / "model.keras").exists() and not (MODEL_DIR / "model_fp16.tflite").exists():
    st.error("No trained model found. Run `make train export` first.")
    st.stop()

name, predict = get_predictor()
upload = st.file_uploader("Upload a soil photo", type=["jpg", "jpeg", "png", "webp"])
if upload:
    raw = upload.getvalue()
    image = load_rgb(io.BytesIO(raw), SIZE)
    st.image(raw, use_container_width=True)
    probs = predict(image)
    top = int(np.argmax(probs))
    if probs[top] < THRESHOLD:
        st.warning(f"Low confidence ({probs[top]:.0%}). The photo may be unclear or not one of the 3 classes.")
    st.subheader(f"{CLASSES[top]} — {probs[top]:.1%}")
    st.bar_chart({c: float(p) for c, p in zip(CLASSES, probs)})
    st.caption(f"Model: {name}")
