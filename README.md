# Water-Segmentation-Project

This project detects and segments **water bodies** from multispectral satellite images using **deep learning models**.  
It includes two **training notebooks** (pretrained and baseline/from-scratch) and a **Flask web app** for deployment.  
The app dynamically selects the best model (pretrained or baseline) based on **validation IoU**.

---

## 📂 Repository Structure
```
Water-Segmentation-Project/
│── WaterSegmentation.ipynb                # Pretrained model training & evaluation
│── WaterSegmentation_baselineModel.ipynb  # Baseline (from-scratch) model training & evaluation
│── app.py                                 # Flask web app for deployment
│── utils.py                               # Utility functions (image reading, feature engineering, etc.)
│── templates/
│   └── index.html                         # HTML template for the Flask app
│── static/
│   └── css/
│       └── styles.css                     # CSS styles for the Flask app
│── best_pretrained_Raw.pth                # Weights for pretrained UNet model
│── best_baseline_model.pth                # Weights for baseline TransUNet model
│── pretrained_model_stats.json            # Metrics + normalization stats (pretrained model)
│── baseline_model_stats.json              # Metrics + normalization stats (baseline model)
│── normalize_stats.json                   # Dataset-wide normalization stats
│── requirements.txt
│── README.md
```

---

## 🧠 Model Overview
Two models are trained and evaluated:

- **Pretrained Model (UNet + ResNet34 encoder)**  
  - Uses raw 12 bands or PCA-reduced 10 bands  
  - Fine-tuned with BCEWithLogits loss  

- **Baseline Model (TransUNet from scratch)**  
  - Encoder–decoder with skip connections  
  - Input: 12 raw bands + 7 engineered features → **19 channels**

The Flask app automatically compares **IoU** from both models and loads the one with better performance.

**Metrics:** IoU, F1, Precision, Recall

---

## 🛰️ Data Preprocessing
- Input: **Multispectral `.tif` images** (12+ bands: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2, …)  
- Masks: **Binary `.png/.tif`** (water=1, non-water=0)  

### Pretrained Model
- Uses 12 bands or 10 PCA-selected bands  
- Normalization: per-channel mean & std  

### Baseline Model
Adds engineered features:
- NDWI, MNDWI, AWEI  
- NDVI  
- Sobel edges (Green)  
- Blue/Red ratio  
- Local variance (Green)  

➡️ Final input = **19-channel tensor**  
➡️ Normalization: dataset-wide mean/std (`normalize_stats.json`)  

**Augmentations:** flips, rotations, brightness/contrast (Albumentations)

---

## ⚙️ Installation
```bash
git clone https://github.com/<your-username>/Water-Segmentation-Project.git
cd Water-Segmentation-Project
pip install -r requirements.txt
```

---

## 🚀 Training (Notebooks)

Open with Jupyter:
```bash
jupyter notebook
```

### Pretrained Notebook (`WaterSegmentation.ipynb`)
1. Load images & masks  
2. (Optional) PCA band selection  
3. Normalize  
4. Train/fine-tune **UNet (ResNet34)**  
5. Save → `best_pretrained_Raw.pth` & `pretrained_model_stats.json`

### Baseline Notebook (`WaterSegmentation_baselineModel.ipynb`)
1. Load images & masks  
2. Compute engineered features  
3. Normalize → `normalize_stats.json`  
4. Train **TransUNet from scratch**  
5. Save → `best_baseline_model.pth` & `baseline_model_stats.json`

---

## 🌐 Deployment (Flask App)

Run the app:
```bash
python app.py
```

Access:
- Local → [http://127.0.0.1:5000](http://127.0.0.1:5000)  
- Network → [http://0.0.0.0:5000](http://0.0.0.0:5000)  

### Features
- **Dynamic model selection** (pretrained vs baseline)  
- Upload `.tif` image (+ optional mask)  
- Automatic preprocessing + feature engineering  
- Predict segmentation mask  
- Visualize:
  - RGB composite  
  - Predicted mask  
  - Ground truth (if uploaded)  

- Confidence scores:
  - Overall probability  
  - Water-only probability  

- Metrics (if mask provided): IoU, F1  

---

## 📊 Example Workflow
1. Upload satellite `.tif` image  
2. Preprocessing & feature extraction  
3. Model predicts water regions  
4. Outputs:
   - Visualizations  
   - Confidence scores  
   - Metrics (if ground truth provided)  

---

## 🔑 Key Files
- **`best_pretrained_Raw.pth`** – UNet pretrained weights  
- **`best_baseline_model.pth`** – TransUNet baseline weights  
- **`pretrained_model_stats.json` / `baseline_model_stats.json`** – Metrics + normalization  
- **`normalize_stats.json`** – Dataset-wide normalization stats  
- **`app.py`** – Flask deployment app  
- **`utils.py`** – Helper functions  
- **`index.html` & `styles.css`** – UI + styling  

---

## 📌 Future Work
- Add REST API for batch predictions  
- Improve mobile responsiveness  
- Integrate live satellite API feeds  
- Explore **ensemble methods** (pretrained + baseline)  

---

✨ Developed by Rowan Yasser  
If you use this repo, please ⭐ star it!  
