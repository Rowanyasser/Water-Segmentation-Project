# Water-Segmentation-Project

This project detects and segments **water bodies** from multispectral satellite images using a **Transformer-UNet (TransUNet)** deep learning model.  
The repository contains both the **training notebook** and a **Streamlit app** for deployment.

---

## 📂 Repository Structure
```
Water-Segmentation-Project/
│── WaterSegmentation.ipynb     # Jupyter Notebook: training, preprocessing & feature engineering
│── model.py                    # TransUNet architecture (CNN + Transformer bottleneck)
│── app.py                      # Streamlit app for deployment
│── final_model.pth             # Trained model weights
│── normalize_stats.json        # Global mean/std for dataset-wide normalization
│── README.md                   # Project documentation
```

---

## 🧠 Model Overview
The model is a **TransUNet**:
- **Encoder (CNN blocks)**: Extract low-level features.
- **Decoder (Upsampling + skip connections)**: Reconstructs segmentation mask.
- **Output**: Binary mask (water vs non-water).

The loss used is a **hybrid of BCE, Dice, and Focal Loss**, balancing precision and recall.

---

## 🛰️ Data Preprocessing
- Input: **Multispectral `.tif` images** (at least 7 bands: Blue, Green, Red, NIR, SWIR1, SWIR2, …).  
- Feature engineering adds extra channels:
  - NDWI (Normalized Difference Water Index)
  - MNDWI (Modified NDWI)
  - AWEI (Automated Water Extraction Index)
  - NDVI (Vegetation Index)
  - Sobel edge filters
- Final input: **22-channel tensor** (bands + engineered features).
- Normalization: Dataset-wide mean & std stored in `normalize_stats.json`.

---

## ⚙️ Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/Water-Segmentation-Project.git
cd Water-Segmentation-Project
pip install -r requirements.txt
```

---

## 🚀 Training (Notebook)
Open the notebook:
```bash
jupyter notebook WaterSegmentation.ipynb
```

Steps inside:
1. Load multispectral images & masks.
2. Compute additional features (NDWI, NDVI, etc.).
3. Normalize dataset (mean/std → `normalize_stats.json`).
4. Train TransUNet model.
5. Save trained model → `final_model.pth`.

---

## 🌐 Deployment (Streamlit App)

Run the app:
```bash
streamlit run app.py
```

### App Features:
- Upload a `.tif` multispectral image.
- Automatic preprocessing + feature engineering.
- Predict **water segmentation mask**.
- Visualize:
  - Original RGB image.
  - Predicted binary mask.
  - Overlay of mask on RGB.
- Download results (`mask.png`, `overlaid_image.png`).
- Water presence detection (reports % of water pixels).

---

## 📊 Example Workflow
1. Upload satellite `.tif` image.
2. App extracts bands & computes features.
3. Model predicts water regions.
4. Output shows if water is present + segmentation mask.

---

## 🔑 Key Files
- **`final_model.pth`** – Trained weights of TransUNet.  
- **`normalize_stats.json`** – Global normalization stats (per-channel mean & std).  
- **`model.py`** – Defines TransUNet model.  
- **`app.py`** – Streamlit deployment app.  

---

## 📌 Future Work
- Support more satellite datasets (Sentinel-2, Landsat, etc.).
- Optimize inference for large images.
- Extend to multi-class segmentation (e.g., water, vegetation, urban).

---

✨ Developed by Rowan Yasser
If you use this repo, please ⭐ star it!
