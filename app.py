import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import io
import base64
import tifffile
from scipy import ndimage
from model import TransUNet
import json

# Config
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path("final_model.pth")
NORMALIZE_STATS_PATH = Path("normalize_stats.json")
NUM_CHANNELS = 22
IMG_SIZE = 256
MIN_BANDS = 7  # Minimum channels for feature engineering
WATER_THRESHOLD = 0.05  # Water presence if >5% of pixels are water

# Feature engineering helpers
def safe_div(a, b, eps=1e-6):
    return a / (b + eps)

def compute_ndwi(green, nir):
    return safe_div(green - nir, green + nir)

def compute_mndwi(green, swir):
    return safe_div(green - swir, green + swir)

def compute_awei(blue, green, nir, swir1, swir2):
    return 4 * (green - swir1) - (0.25 * nir + 2.75 * swir2)

def compute_ndvi(red, nir):
    return safe_div(nir - red, nir + red)

# Load model
@st.cache_resource
def load_model():
    try:
        model = TransUNet(in_ch=NUM_CHANNELS, n_classes=1).to(DEVICE)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval()
        st.success("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()
if model is None:
    st.stop()

# Load normalize_stats
try:
    normalize_stats = json.load(open(NORMALIZE_STATS_PATH))
    normalize_stats = {'mean': np.array(normalize_stats['mean'], dtype=np.float32),
                       'std': np.array(normalize_stats['std'], dtype=np.float32)}
    st.success("Loaded normalize_stats successfully")
except Exception as e:
    st.error(f"Failed to load normalize_stats: {e}. Please run notebook cell to save normalize_stats.json.")
    st.stop()

# Preprocessing
def preprocess_image(uploaded_file):
    try:
        img_bytes = uploaded_file.read()
        img = tifffile.imread(io.BytesIO(img_bytes))
        if img.ndim == 3 and img.shape[0] > img.shape[2]:
            img = img.transpose(2, 0, 1)
        elif img.ndim == 2:
            img = img[None, ...]
    except Exception as e:
        st.error(f"Failed to load TIFF: {e}. This app requires a multispectral .tif file.")
        return None, None, None
    img = img.astype(np.float32)

    C, H, W = img.shape
    st.write(f"Input image shape: {C, H, W}")

    # Check for sufficient channels
    if C < MIN_BANDS:
        st.error(f"Image has {C} channels, but at least {MIN_BANDS} are required for feature engineering (e.g., blue, green, red, nir, swir1, swir2).")
        return None, None, None

    # Feature engineering
    fi = {'blue': 0, 'green': 1, 'red': 2, 'nir': 4, 'swir1': 5, 'swir2': 6}
    extra_chs = []
    try:
        extra_chs.append(compute_ndwi(img[fi['green']], img[fi['nir']])[None, ...].astype(np.float32))
        extra_chs.append(compute_mndwi(img[fi['green']], img[fi['swir1']])[None, ...].astype(np.float32))
        extra_chs.append(compute_awei(img[fi['blue']], img[fi['green']], img[fi['nir']],
                                     img[fi['swir1']], img[fi['swir2']])[None, ...].astype(np.float32))
        extra_chs.append(compute_ndvi(img[fi['red']], img[fi['nir']])[None, ...].astype(np.float32))
        sobel_edges = np.stack([ndimage.sobel(img[i], mode='constant') for i in range(C)], axis=0).astype(np.float32)
        extra_chs.append(sobel_edges.mean(axis=0)[None, ...])
    except Exception as e:
        st.error(f"Feature engineering failed: {e}")
        return None, None, None

    if extra_chs:
        img = np.concatenate([img] + extra_chs, axis=0)

    if img.shape[0] < NUM_CHANNELS:
        img = np.pad(img, ((0, NUM_CHANNELS - img.shape[0]), (0, 0), (0, 0)), mode='constant').astype(np.float32)
    assert img.shape[0] == NUM_CHANNELS, f"Channels mismatch: {img.shape[0]} != {NUM_CHANNELS}"

    img_resized = np.zeros((NUM_CHANNELS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for ch in range(NUM_CHANNELS):
        img_resized[ch] = cv2.resize(img[ch], (IMG_SIZE, IMG_SIZE))

    mean = normalize_stats['mean'][:, None, None]
    std = normalize_stats['std'][:, None, None]
    if mean.shape[0] < img_resized.shape[0]:
        extra = img_resized.shape[0] - mean.shape[0]
        mean = np.concatenate([mean, np.zeros((extra, 1, 1), dtype=np.float32)], axis=0)
        std = np.concatenate([std, np.ones((extra, 1, 1), dtype=np.float32)], axis=0)
    img_norm = (img_resized - mean) / (std + 1e-6)

    tensor = torch.from_numpy(img_norm).unsqueeze(0).to(DEVICE).float()

    # Create RGB for display
    try:
        img_rgb = img[[2, 1, 0], :, :].transpose(1, 2, 0)  # RGB bands
        img_rgb = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))  # Resize to 256x256
        img_rgb = (img_rgb - img_rgb.min()) / (img_rgb.max() - img_rgb.min() + 1e-6) * 255
        img_rgb = img_rgb.astype(np.uint8)
        st.write(f"RGB image shape: {img_rgb.shape}")
    except Exception as e:
        st.error(f"Failed to create RGB image for display: {e}")
        img_rgb = None

    return tensor, img_resized.transpose(1, 2, 0), img_rgb

# Prediction
def predict_mask(image_tensor, threshold=0.5):
    try:
        with torch.no_grad():
            logits = model(image_tensor)
            probs = torch.sigmoid(logits)
            mask = (probs > threshold).float().cpu().numpy()[0, 0]
        st.write(f"Mask shape: {mask.shape}")
        return mask
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Overlay mask
def overlay_mask(img_rgb, mask, alpha=0.5):
    try:
        mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))  # Ensure mask matches img_rgb
        mask_colored = np.zeros_like(img_rgb)  # Shape (256, 256, 3)
        mask_colored[mask_resized > 0] = [255, 0, 0]
        overlaid = cv2.addWeighted(img_rgb, 1 - alpha, mask_colored, alpha, 0)
        return overlaid
    except Exception as e:
        st.error(f"Overlay failed: {e}")
        return None

# Streamlit UI
st.title("Advanced Water Body Segmentation App")
st.write("Upload a multispectral satellite image (.tif, with at least 7 bands: blue, green, red, nir, swir1, swir2, etc.) to segment water bodies.")

uploaded_file = st.file_uploader("Upload an image (.tif)", type=["tif"])

if uploaded_file is not None:
    tensor, img_resized, img_rgb = preprocess_image(uploaded_file)
    if tensor is None or img_resized is None or img_rgb is None:
        st.error("Preprocessing failed. Check logs above.")
        st.stop()

    st.image(img_rgb, caption="Uploaded Image (RGB)", use_column_width=True)

    threshold = st.slider("Confidence Threshold", min_value=0.1, max_value=0.9, value=0.5, step=0.05)
    alpha = st.slider("Mask Transparency (for overlay)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

    mask = predict_mask(tensor, threshold)
    if mask is None:
        st.stop()

    # Water presence check
    water_ratio = np.mean(mask > 0)
    if water_ratio > WATER_THRESHOLD:
        st.success(f"Water Present: Yes (Area: {water_ratio * 100:.2f}%)")
    else:
        st.warning(f"Water Present: No (Area: {water_ratio * 100:.2f}%)")

    overlaid = overlay_mask(img_rgb, mask, alpha)
    if overlaid is None:
        st.stop()

    st.subheader("Prediction Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(img_rgb, caption="Original Image (RGB)", use_column_width=True)
    with col2:
        st.image((mask * 255).astype(np.uint8), caption="Predicted Water Mask", use_column_width=True)
    with col3:
        st.image(overlaid, caption="Overlaid Mask", use_column_width=True)

    st.subheader("Downloads")
    def get_download_link(img, filename, text):
        try:
            buffered = io.BytesIO()
            Image.fromarray(img).save(buffered, format="PNG")
            b64 = base64.b64encode(buffered.getvalue()).decode()
            return f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
        except Exception as e:
            st.error(f"Download link creation failed: {e}")
            return ""

    st.markdown(get_download_link((mask * 255).astype(np.uint8), "water_mask.png", "Download Mask"), unsafe_allow_html=True)
    st.markdown(get_download_link(overlaid, "overlaid_image.png", "Download Overlaid Image"), unsafe_allow_html=True)
