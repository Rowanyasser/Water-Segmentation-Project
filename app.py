from flask import Flask, request, render_template
import torch
import torch.nn as nn
import numpy as np
import io
import base64
import json
import matplotlib.pyplot as plt
from PIL import Image
import logging
import segmentation_models_pytorch as smp
import os
import torch.nn.functional as F
from utils import read_multiband_tif, read_mask, safe_div, compute_ndwi, compute_mndwi, compute_awei, compute_ndvi, sobel, ndimage
from sklearn.metrics import f1_score, jaccard_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dynamically select the best model based on IoU
try:
    with open('pretrained_model_stats.json', 'r') as f_pre:
        pretrained_stats = json.load(f_pre)
    with open('baseline_model_stats.json', 'r') as f_base:
        baseline_stats = json.load(f_base)
    pretrained_iou = pretrained_stats['metrics']['IoU']
    baseline_iou = baseline_stats['metrics']['IoU']
    USE_PRETRAINED = pretrained_iou > baseline_iou
    BEST_MODEL_PATH = "best_pretrained_Raw.pth" if USE_PRETRAINED else "best_baseline_model.pth"
    IN_CHANNELS = 3 if USE_PRETRAINED else 19  # Adjusted to 19 for baseline to match checkpoint
    IS_PRETRAINED = USE_PRETRAINED
    STATS_FILE = "pretrained_model_stats.json" if USE_PRETRAINED else "baseline_model_stats.json"
    logging.info(f"Selected model: {'Pretrained' if USE_PRETRAINED else 'Baseline (from scratch)'} with IoU {max(pretrained_iou, baseline_iou):.4f}")
except FileNotFoundError as e:
    logging.error(f"Model stats file not found: {e}")
    raise

# Load stats for the selected model
try:
    with open(STATS_FILE, 'r') as f:
        stats = json.load(f)
    mean = np.array(stats['normalize_stats']['mean'])
    std = np.array(stats['normalize_stats']['std'])
    feature_indices = stats.get('feature_indices', {})  # For baseline
    selected_bands = stats.get('selected_bands', None)  # For pretrained
except FileNotFoundError as e:
    logging.error(f"Stats file not found: {e}")
    raise

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )
    def forward(self, x): return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch, dropout=dropout)
        )
    def forward(self, x): return self.mpconv(x)

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dropout=0.3):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_ch, out_ch, dropout=dropout)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, in_ch=19, n_classes=1, base_c=32, bilinear=True, dropout=0.3):
        super().__init__()
        self.inc = DoubleConv(in_ch, base_c, dropout=dropout)
        self.down1 = Down(base_c, base_c*2, dropout=dropout)
        self.down2 = Down(base_c*2, base_c*4, dropout=dropout)
        self.down3 = Down(base_c*4, base_c*8, dropout=dropout)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c*8, base_c*16 // factor, dropout=dropout)
        self.up1 = Up(base_c*16, base_c*8 // factor, bilinear, dropout=dropout)
        self.up2 = Up(base_c*8, base_c*4 // factor, bilinear, dropout=dropout)
        self.up3 = Up(base_c*4, base_c*2 // factor, bilinear, dropout=dropout)
        self.up4 = Up(base_c*2, base_c, bilinear, dropout=dropout)
        self.outc = nn.Conv2d(base_c, n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

# Load model with error handling
if IS_PRETRAINED:
    model = smp.UNet(encoder_name="resnet34", in_channels=IN_CHANNELS, classes=1)  # Use UNet to match checkpoint structure
else:
    model = TransUNet(in_ch=IN_CHANNELS, n_classes=1)  # Use baseline TransUNet
try:
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
except RuntimeError as e:
    logging.error(f"Failed to load state dict: {e}. Ensure the model architecture matches the checkpoint.")
    raise
model.eval()
model = model.to(DEVICE)
logging.info(f"Loaded model from {BEST_MODEL_PATH} with {IN_CHANNELS} input channels")

app = Flask(__name__, template_folder='templates', static_folder='static')

def predict_image(image_file, mask_file=None):
    """
    Returns:
        viz_data (base64 png),
        prob_map (uint8 0-255),
        overall_conf (float 0..1),
        pred_conf (float 0..1),
        metrics (dict or None) -> {"IoU":..., "F1":...}
    """
    # Define temp_path for the image file
    temp_path = f"/tmp/{image_file.filename}"
    try:
        # Save uploaded image file
        image_file.save(temp_path)
        # Read the multi-band image
        img = read_multiband_tif(temp_path)
        if img is None or img.size == 0:
            raise ValueError("Failed to read the image file or image is empty.")
        C, H, W = img.shape

        # Feature engineering (only if selected_bands is None, i.e., Raw dataset)
        extra_chs = []
        fi = feature_indices if not IS_PRETRAINED else {i: i for i in range(C)}
        if selected_bands and IS_PRETRAINED:
            img = img[selected_bands]
        if 'green' in fi and 'nir' in fi:
            extra_chs.append(compute_ndwi(img[fi['green']], img[fi['nir']])[None, ...])
        if 'green' in fi and 'swir1' in fi:
            extra_chs.append(compute_mndwi(img[fi['green']], img[fi['swir1']])[None, ...])
        if all(k in fi for k in ('coastal', 'green', 'nir', 'swir1', 'swir2')):
            extra_chs.append(compute_awei(img[fi['coastal']], img[fi['green']], img[fi['nir']], img[fi['swir1']], img[fi['swir2']])[None, ...])
        if 'red' in fi and 'nir' in fi:
            extra_chs.append(compute_ndvi(img[fi['red']], img[fi['nir']])[None, ...])
        if 'green' in fi:
            edges = np.sqrt(sobel(img[fi['green']], axis=0)**2 + sobel(img[fi['green']], axis=1)**2)
            extra_chs.append(edges[None, ...])
        if 'blue' in fi and 'red' in fi:
            extra_chs.append(safe_div(img[fi['blue']], img[fi['red']])[None, ...])
        if 'green' in fi:
            local_mean = ndimage.uniform_filter(img[fi['green']], size=5)
            local_sqr_mean = ndimage.uniform_filter(img[fi['green']]**2, size=5)
            local_var = local_sqr_mean - local_mean**2
            extra_chs.append(local_var[None, ...])

        if extra_chs:
            img = np.concatenate([img] + extra_chs, axis=0)

        # Normalize
        if mean.shape[0] != img.shape[0]:
            extra = img.shape[0] - mean.shape[0]
            mean_ext = np.concatenate([mean, np.zeros(extra)])
            std_ext = np.concatenate([std, np.ones(extra)])
        else:
            mean_ext, std_ext = mean, std
        img_norm = (img - mean_ext[:, None, None]) / (std_ext[:, None, None] + 1e-6)

        # Predict
        img_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            output = torch.sigmoid(model(img_tensor)).squeeze().cpu().numpy()
        if output.ndim == 3 and output.shape[0] == 1:
            output = output[0]

        # Binary mask and probability map
        pred_mask_binary = (output > 0.5).astype(np.uint8)
        pred_mask = pred_mask_binary * 255
        prob_map = (output * 255).astype(np.uint8)

        # Confidence metrics
        overall_conf = float(output.mean())
        pred_conf = float(output[pred_mask_binary == 1].mean()) if pred_mask_binary.sum() > 0 else 0.0

        # RGB composite for visualization
        if img_norm.shape[0] >= 3:
            rgb_unnorm = (img_norm[:3] * std_ext[:3, None, None]) + mean_ext[:3, None, None]
            rgb_img_arr = np.clip(rgb_unnorm.transpose(1, 2, 0), 0, 1) * 255.0
            rgb_img = Image.fromarray(rgb_img_arr.astype(np.uint8))
        else:
            single = img_norm[0]
            rgb_img_arr = np.clip((single - single.min()) / (single.max() - single.min() + 1e-6), 0, 1)
            rgb_img_arr = np.stack([rgb_img_arr]*3, axis=-1) * 255.0
            rgb_img = Image.fromarray(rgb_img_arr.astype(np.uint8))

        # Ground truth metrics if mask_file is provided
        metrics = None
        gt_mask = None
        if mask_file and mask_file.filename:
            mask_temp_path = f"/tmp/{mask_file.filename}"
            try:
                mask_file.save(mask_temp_path)
                gt_mask = read_mask(mask_temp_path)
                if gt_mask.shape != pred_mask_binary.shape:
                    gt_pil = Image.fromarray((gt_mask * 255).astype(np.uint8)).convert("L")
                    gt_pil = gt_pil.resize((pred_mask_binary.shape[1], pred_mask_binary.shape[0]), resample=Image.NEAREST)
                    gt_mask = (np.array(gt_pil) > 127).astype(np.uint8)

                try:
                    iou = jaccard_score(gt_mask.flatten(), pred_mask_binary.flatten(), zero_division=0)
                    f1 = f1_score(gt_mask.flatten(), pred_mask_binary.flatten(), zero_division=0)
                    metrics = {"IoU": float(iou), "F1": float(f1)}
                except Exception as e:
                    logging.warning(f"Metrics computation failed: {e}")
                    metrics = {"IoU": 0.0, "F1": 0.0}
            except Exception as e:
                logging.error(f"Failed to process mask file: {e}")
                metrics = None
            finally:
                if os.path.exists(mask_temp_path):
                    try:
                        os.remove(mask_temp_path)
                    except Exception as e:
                        logging.warning(f"Failed to remove temporary mask file: {e}")

        # Visualization
        fig, axs = plt.subplots(1, 3 if gt_mask is not None else 2, figsize=(12, 4))
        axs[0].imshow(rgb_img)
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        axs[1].imshow(pred_mask, cmap='gray')
        axs[1].set_title('Predicted Mask')
        axs[1].axis('off')
        if gt_mask is not None:
            axs[2].imshow(gt_mask * 255, cmap='gray')
            axs[2].set_title('Ground Truth')
            axs[2].axis('off')
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        viz_data = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return viz_data, prob_map, overall_conf, pred_conf, metrics

    finally:
        # Clean up temporary image file
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as e:
                logging.warning(f"Failed to remove temporary image file: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    viz_data = None
    error = None
    overall_conf, pred_conf, metrics = None, None, None

    if request.method == 'POST':
        image_file = request.files.get('image')
        mask_file = request.files.get('mask')
        if image_file and image_file.filename:
            try:
                viz_data, _, overall_conf, pred_conf, metrics = predict_image(image_file, mask_file)
            except Exception as e:
                logging.error(f"Prediction error: {e}")
                error = f"Error: {str(e)}"
        else:
            error = "No valid image file provided."

    best_iou = max(pretrained_iou, baseline_iou)

    return render_template(
        'index.html',
        viz_data=viz_data,
        error=error,
        pretrained_iou=pretrained_iou,
        baseline_iou=baseline_iou,
        best_iou=best_iou,
        use_pretrained=USE_PRETRAINED,
        overall_conf=overall_conf,
        pred_conf=pred_conf,
        metrics=metrics
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)