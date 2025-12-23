# app/app.py
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import streamlit as st
from PIL import Image
import pydicom
import torch

# Make src importable when running `streamlit run app/app.py`
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import BEST_MODEL_PATH, IMAGE_SIZE
from src.models.alznet import AlzNet
from src.data.dataset import get_transforms, CLASS_MAP


@st.cache_resource
def load_model(device: torch.device):
    model = AlzNet(num_classes=len(CLASS_MAP))
    if not BEST_MODEL_PATH.exists():
        st.error(f"Model checkpoint not found at {BEST_MODEL_PATH}. Train the model first.")
        return None

    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def read_image(file) -> Image.Image:
    """
    Accepts uploaded file (DICOM or image) and returns a PIL grayscale image.
    """
    suffix = Path(file.name).suffix.lower()
    if suffix == ".dcm":
        ds = pydicom.dcmread(file)
        arr = ds.pixel_array.astype(np.float32)
        arr -= arr.min()
        if arr.max() > 0:
            arr /= arr.max()
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr).convert("L")
    else:
        img = Image.open(file).convert("L")
    return img


def predict(image: Image.Image, model: AlzNet, device: torch.device) -> Tuple[str, float, np.ndarray]:
    transform = get_transforms(train=False)
    img_t = transform(image).unsqueeze(0).to(device)  # (1, 1, H, W)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    class_idx = int(np.argmax(probs))
    inv_class_map = {v: k for k, v in CLASS_MAP.items()}
    predicted_class = inv_class_map[class_idx]
    confidence = float(probs[class_idx])
    return predicted_class, confidence, probs


def main():
    st.set_page_config(page_title="Alzheimer's MRI Classifier", page_icon="🧠", layout="centered")

    st.title("🧠 Early Detection of Alzheimer's Disease from MRI")
    st.write(
        "Upload a brain MRI slice (DICOM or image), and the deep learning model "
        "will classify it as **Normal (NC)**, **Mild Cognitive Impairment (MCI)**, or **Alzheimer's Disease (AD)**."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)
    if model is None:
        return

    uploaded_file = st.file_uploader("Upload MRI file (.dcm, .png, .jpg, .jpeg)", type=["dcm", "png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Show original image
        img = read_image(uploaded_file)
        st.image(img.resize(IMAGE_SIZE), caption="Uploaded MRI slice (resized)", use_column_width=False)

        if st.button("Run Prediction"):
            with st.spinner("Analyzing MRI..."):
                pred_class, conf, probs = predict(img, model, device)

            st.markdown(f"### ✅ Prediction: **{pred_class}**")
            st.write(f"Confidence: **{conf * 100:.2f}%**")

            # Show per-class probabilities
            st.subheader("Class probabilities")
            inv_class_map = {v: k for k, v in CLASS_MAP.items()}
            for idx, p in enumerate(probs):
                st.write(f"{inv_class_map[idx]}: {p * 100:.2f}%")


if __name__ == "__main__":
    main()
