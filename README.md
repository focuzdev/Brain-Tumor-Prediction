<div align="center">
  
# 🧠 NeuroScan AI 
### Brain Tumor MRI Classification with Explainable AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://hafeez-brain-tumor-prediction.streamlit.app/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Live Demo:** [https://hafeez-brain-tumor-prediction.streamlit.app/](https://hafeez-brain-tumor-prediction.streamlit.app/)

</div>

---

## 📌 Overview

NeuroScan AI is an interactive clinical decision support tool that uses deep learning to classify brain MRI scans into **four categories**:

| Class | Description |
|-------|-------------|
| 🧠 **Glioma** | High-grade brain tumor with irregular margins |
| 🔵 **Meningioma** | Usually benign, extra-axial mass |
| 🟣 **Pituitary Tumor** | Sellar region tumor, often benign |
| 🟢 **No Tumor** | Normal brain parenchyma |

### 🔥 Key Features

- ✅ **Deep Learning Classification** - EfficientNetB0 CNN model
- ✅ **Grad-CAM Explainability** - Color-coded heatmaps showing where the model looks (Red/Yellow = High Attention)
- ✅ **AI-Generated Clinical Reports** - Powered by Anthropic Claude API
- ✅ **Interactive UI** - Upload MRI or test with sample images
- ✅ **Download Reports** - Export JSON results

---

## 🚀 Live Demo

**Test the application now:** 👉 [https://hafeez-brain-tumor-prediction.streamlit.app/](https://hafeez-brain-tumor-prediction.streamlit.app/)

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
