Water-Segmentation-Project
This project detects and segments water bodies from multispectral satellite images using deep learning models.It includes two training notebooks (pretrained and baseline/from-scratch) and a Flask web app for deployment. The app dynamically selects the best model (pretrained or baseline) based on validation IoU.

📂 Repository Structure
Water-Segmentation-Project/
│── WaterSegmentation.ipynb                # Jupyter Notebook: Pretrained model training & evaluation
│── WaterSegmentation_baselineModel.ipynb  # Jupyter Notebook: Baseline (from-scratch) model training & evaluation
│── app.py                                 # Flask web app for deployment
│── utils.py                               # Utility functions (image reading, feature engineering, etc.)
│── templates/
│   └── index.html                         # HTML template for the Flask app
│── static/
│   └── css/
│       └── styles.css                     # CSS styles for the Flask app
│── best_pretrained_Raw.pth                # Trained weights for the best pretrained model
│── best_baseline_model.pth                # Trained weights for the baseline model
│── pretrained_model_stats.json            # Stats (metrics, normalization) for pretrained model
│── baseline_model_stats.json              # Stats (metrics, normalization) for baseline model
│── normalize_stats.json                   # Global mean/std for dataset-wide normalization (from baseline)
│── requirements.txt                       # Python dependencies
│── README.md                              # Project documentation


🧠 Model Overview
Two models are trained and evaluated:

Pretrained Model: UNet with ResNet34 encoder (from segmentation_models_pytorch). Uses raw bands (12 channels) or chosen bands (10 channels with PCA). Fine-tuned on water segmentation.
Baseline Model (From Scratch): Custom TransUNet (CNN encoder + upsampling decoder with skip connections). Uses 12 raw bands + 7 engineered features (total 19 channels).

The Flask app compares IoU from pretrained_model_stats.json and baseline_model_stats.json, then loads the better model dynamically.Loss: BCEWithLogits (with optional positive weighting for class imbalance).Metrics: IoU (Jaccard), F1, Precision, Recall.

🛰️ Data Preprocessing

Input: Multispectral .tif images (12+ bands: Coastal, Blue, Green, Red, NIR, SWIR1, SWIR2, etc.). Masks: Binary .png/.tif (water=1, non-water=0).
Pretrained Model:
Uses raw (12 bands) or chosen bands (10 with PCA for dimensionality reduction).
Normalization: Per-channel mean/std from dataset stats.


Baseline Model:
Feature engineering adds extra channels:
NDWI (Normalized Difference Water Index)
MNDWI (Modified NDWI)
AWEI (Automated Water Extraction Index)
NDVI (Vegetation Index)
Sobel edges (on Green band)
Blue/Red ratio
Local variance (on Green band)


Final input: 19-channel tensor (bands + features).
Normalization: Dataset-wide mean & std stored in normalize_stats.json.


Augmentations (in notebooks): Albumentations (flips, rotations, brightness/contrast).


⚙️ Installation
Clone the repo and install dependencies:
git clone https://github.com/<your-username>/Water-Segmentation-Project.git
cd Water-Segmentation-Project
pip install -r requirements.txt


🚀 Training (Notebooks)
Open the notebooks:
jupyter notebook

Pretrained Notebook (WaterSegmentation.ipynb):

Load images & masks.
Optional: Select bands + PCA.
Normalize using computed stats.
Train/fine-tune UNet (ResNet34 encoder).
Evaluate (IoU, F1, etc.) & save model/stats → best_pretrained_Raw.pth & pretrained_model_stats.json.

Baseline Notebook (WaterSegmentation_baselineModel.ipynb):

Load images & masks.
Compute features (NDWI, edges, etc.).
Normalize → normalize_stats.json.
Train TransUNet from scratch.
Evaluate & save model/stats → best_baseline_model.pth & baseline_model_stats.json.


🌐 Deployment (Flask App)
Run the app:
python app.py

Access at http://127.0.0.1:5000 (or http://0.0.0.0:5000 for network access).
App Features:

Dynamic Model Selection: Loads the model (pretrained or baseline) with the highest IoU.
Upload a .tif multispectral image (required) and optional ground truth mask (.png/.tif).
Automatic preprocessing + feature engineering (based on selected model).
Predict water segmentation mask.
Visualize (side-by-side):
Original RGB composite.
Predicted binary mask.
Ground truth mask (if uploaded).


Confidence Scores:
Overall confidence (average probability across image).
Predicted mask confidence (average probability on water pixels).


Metrics (if ground truth uploaded): IoU and F1.
Responsive UI with Bootstrap, custom CSS (gradient backgrounds, animations, hover effects).


📊 Example Workflow

Upload satellite .tif image (and optional mask).
App preprocesses (bands selection/features + normalization).
Model predicts water regions.
Output: Visualization, confidences, and metrics (if mask provided).


🔑 Key Files

best_pretrained_Raw.pth – Weights for pretrained UNet model.  
best_baseline_model.pth – Weights for baseline TransUNet model.  
pretrained_model_stats.json / baseline_model_stats.json – Metrics (IoU/F1), normalization stats, band/feature indices.  
normalize_stats.json – Additional normalization stats (from baseline).  
app.py – Flask deployment app (handles uploads, predictions, rendering).  
utils.py – Helper functions for reading images/masks and feature engineering.  
index.html & styles.css – UI template and styles (Bootstrap + custom animations).


📌 Future Work

Add API endpoints for batch predictions.
Optimize for mobile/responsive design.
Integrate real-time satellite data feeds (e.g., via APIs).
Explore ensemble models (combine pretrained + baseline).


✨ Developed by Rowan YasserIf you use this repo, please ⭐ star it!
