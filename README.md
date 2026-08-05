<div align="center">
  
# 🧠 NeuroScan AI
### Explainable Brain Tumor MRI Classification using Deep Learning

<p align="center">

### 🚀 **Try the Live Application**

**👉 https://brain-tumor-prediction-mri-classifier.streamlit.app/**

</p>

---

## 📌 Overview

NeuroScan AI is an interactive clinical decision support application that uses deep learning and Explainable AI (Grad-CAM) to classify brain MRI scans into four diagnostic categories while providing visual explanations and AI-assisted clinical reporting.

| Class | Description |
|-------|-------------|
| 🧠 **Glioma** | High-grade brain tumor with irregular margins |
| 🔵 **Meningioma** | Usually benign, extra-axial mass |
| 🟣 **Pituitary Tumor** | Sellar region tumor, often benign |
| 🟢 **No Tumor** | Normal brain parenchyma |

## 🔥 Key Features

- ✅ **Deep Learning Classification** using EfficientNetB0
- ✅ **Grad-CAM Explainability** with visual attention heatmaps
- ✅ **AI-Generated Clinical Reports** powered by Anthropic Claude
- ✅ **Interactive Web Interface** with image upload and sample MRI scans
- ✅ **Downloadable Prediction Reports** in JSON format

---

## 🚀 Live Demo

**Launch NeuroScan AI:**

### 👉 https://brain-tumor-prediction-mri-classifier.streamlit.app/

### Quick Test Steps:
1. Select a sample image from the sidebar (Glioma, Meningioma, etc.)
2. Click **"Analyze Scan"**
3. View the prediction, Grad-CAM heatmap, and clinical report

---
## 📊 How It Works
┌─────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ MRI Scan │────▶│ EfficientNetB0 │────▶│ Prediction │
│ (Input) │ │ (CNN Model) │ │ + Confidence │
└─────────────┘ └─────────────────┘ └────────┬────────┘
│
▼

┌─────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ Clinical │◀────│ Claude AI │◀────│ Grad-CAM │
│ Report │ │ (Text Gen) │ │ (Heatmap) │
└─────────────┘ └─────────────────┘ └─────────────────┘


### Grad-CAM Visualization

| Original MRI | Grad-CAM Overlay | Interpretation |
|--------------|------------------|----------------|
| ![Original](https://via.placeholder.com/150?text=MRI) | ![Grad-CAM](https://via.placeholder.com/150?text=Heatmap) | Red/Yellow = High attention |

---

## 📁 Project Structure
NeuroScan-AI/
├── streamlit_app.py # Main application
├── requirements.txt # Dependencies
├── brain_tumor_model.h5 # Pre-trained CNN model
├── samples/ # Sample MRI images
│ ├── glioma.jpg
│ ├── meningioma.jpg
│ ├── pituitary.jpg
│ └── no_tumor.jpg
└── README.md # This file


---

## 🛠️ Local Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/focuzguy1/Brain-Tumor-Prediction.git
cd Brain-Tumor-Prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run streamlit_app.py

## 📊 How It Works
