import os
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import json

# -------------------------
# Config
# -------------------------
MODEL_PATH = r"C:\Users\Neeraj Tiwari\Desktop\Medical Imaging\model\medical_cnn.h5"
CLASS_MAP_PATH = r"C:\Users\Neeraj Tiwari\Desktop\Medical Imaging\model\class_indices.json"

# -------------------------
# Load model
# -------------------------
@st.cache_resource
def load_model_and_metadata():
    model = keras.models.load_model(MODEL_PATH, compile=False)
    input_shape = model.input_shape  # (None, H, W, C)
    output_shape = model.output_shape  # (None, num_classes) or (None, 1)
    num_classes = output_shape[-1]
    in_h, in_w, in_c = input_shape[1], input_shape[2], input_shape[3]

    # Try to load class map if present
    if os.path.exists(CLASS_MAP_PATH):
        try:
            with open(CLASS_MAP_PATH, "r") as f:
                class_map = json.load(f)
        except Exception:
            class_map = None
    else:
        class_map = None

    if class_map is None:
        # Default for binary classification
        class_names = ["Normal", "Pneumonia"] if num_classes == 2 else [f"class_{i}" for i in range(num_classes)]
    else:
        # Ensure order matches indices
        class_names = [None] * len(class_map)
        for name, idx in class_map.items():
            if 0 <= idx < len(class_map):
                class_names[idx] = name
        for i in range(len(class_names)):
            if class_names[i] is None:
                class_names[i] = f"class_{i}"

    return model, (in_h, in_w, in_c), num_classes, class_names

model, (IN_H, IN_W, IN_C), NUM_CLASSES, CLASS_NAMES = load_model_and_metadata()

# -------------------------
# Preprocess
# -------------------------
def preprocess_image(pil_img):
    if IN_C == 1:
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")

    pil_img = pil_img.resize((IN_W, IN_H))
    arr = np.array(pil_img).astype("float32") / 255.0

    if IN_C == 1 and arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)

    arr = np.expand_dims(arr, axis=0)
    return arr

# -------------------------
# Predict
# -------------------------
def predict(img_batch):
    raw = model.predict(img_batch, verbose=0)

    if NUM_CLASSES == 1:  # Sigmoid binary
        p = float(raw[0][0])
        probs = np.array([1.0 - p, p])
    else:  # Softmax multi-class
        probs = raw[0]
        probs = probs / np.sum(probs)

    return {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)}

# -------------------------
# UI
# -------------------------
st.title("AI Medical Imaging Diagnosis (CNN)")
st.caption("Educational demo — not medical advice.")

with st.sidebar:
    uploaded_image = st.file_uploader("Upload X-ray / image", type=["jpg", "jpeg", "png"])
    st.markdown(f"**Model input:** {IN_H}×{IN_W}×{IN_C}")
    st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")

def predict_proba(img_batch):
    preds = model.predict(img_batch)  # Shape (1,1)
    prob_pneumonia = float(preds[0][0])  # single probability
    prob_normal = 1 - prob_pneumonia     # complementary probability

    return {
        "Normal": prob_normal,
        "Pneumonia": prob_pneumonia
    }

# Define your class labels (order must match your training model)
CLASS_NAMES = ["Normal", "Pneumonia"]

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded X-ray", use_container_width=True)

    batch = preprocess_image(image)
    probs = predict_proba(batch)

    # Show probabilities
    st.subheader("Predicted Probabilities")
    for k, v in probs.items():
        st.write(f"- **{k}**: {v:.3f}")

    # Final prediction
    predicted_class = max(probs, key=probs.get)
    confidence = probs[predicted_class]

    st.subheader("Final Prediction")
    st.write(f"**Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}")
