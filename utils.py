import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import sobel
from scipy import ndimage

def read_multiband_tif(path):
    arr = tifffile.imread(str(path))
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] <= 12 and arr.shape[0] != arr.shape[2]:
        return arr  # (C,H,W)
    if arr.ndim == 3:
        arr = np.moveaxis(arr, -1, 0)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr

def read_mask(path):
    mask = tifffile.imread(str(path)) if str(path).endswith(".tif") else np.array(Image.open(path))
    mask = (mask > 0).astype(np.uint8)
    return mask

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