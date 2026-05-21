# NeuroScan AI

**Brain Tumor MRI Classification and Explainability System**

A clinical-grade web application that classifies brain tumor MRI scans into four categories using a ResNet50V2 ensemble CNN, generates Grad-CAM explainability heatmaps showing which regions drove the prediction, and produces detailed AI-written clinical reports via the Anthropic Claude API.

Built with Streamlit. Intended as a decision-support research tool and not a substitute for professional medical diagnosis.
Demo: https://brain-tumor-prediction-mri-classifier.streamlit.app/

---

## Screenshots

**Dashboard — Dark Mode**

<!-- Paste your screenshot here. Recommended: full-page capture at 1440px width -->
<!-- To add: drag the image into your repo's /docs or /assets folder, then update the path below -->
![Dashboard Dark Mode](docs/screenshots/dashboard_dark.png)

---

**Dashboard — Light Mode**

<!-- Paste your screenshot here -->
![Dashboard Light Mode](docs/screenshots/dashboard_light.png)

---

**Grad-CAM Heatmap Output**

<!-- Recommended: capture the 4-panel figure (Original / Overlay / Activation Map / Histogram) -->
![Grad-CAM Heatmap](docs/screenshots/gradcam_output.png)

---

**AI Clinical Report**

<!-- Capture the report tabs: Clinical Interpretation, Location, Patient Summary, Reliability -->
![Clinical Report](docs/screenshots/clinical_report.png)

---

**Model Performance Panel**

<!-- Capture the training history curves and ensemble confusion matrix -->
![Model Performance](docs/screenshots/model_performance.png)

---

## Features

- **4-Class CNN Classification** — Glioma, Meningioma, Pituitary Tumor, No Tumor
- **Grad-CAM Explainability** — Real CNN gradients via TensorFlow GradientTape; synthetic fallback in demo mode
- **Ensemble Architecture** — ResNet50V2 + MobileNetV2 soft-voting, 95.31% accuracy
- **AI Clinical Report** — Claude Sonnet generates structured clinical interpretation, risk level, next steps, and reliability score per scan
- **Interactive Heatmap Viewer** — Overlay, pure activation map, histogram, and colorscale legend
- **Export** — Full JSON report with prediction, confidence, Grad-CAM stats, and AI narrative
- **Light / Dark Mode** — Theme toggle in the navigation bar
- **Demo Mode** — Runs without the model file using pre-defined predictions and MRI-derived synthetic heatmaps

---

## Architecture

```
Upload MRI  →  ResNet50V2 + MobileNetV2  →  Softmax Ensemble
                                                    ↓
                                        Grad-CAM (GradientTape)
                                                    ↓
                                        Claude AI Clinical Report
                                                    ↓
                                        JSON Export
```

### Model Details

| Property | Value |
|---|---|
| Primary model | ResNet50V2 |
| Ensemble | + MobileNetV2 (soft-vote) |
| Input size | 224 × 224 RGB |
| Preprocessing | `resnet_v2.preprocess_input` |
| Classes | 4 |
| Ensemble accuracy | 95.31% |
| XAI method | Grad-CAM via GradientTape |

### Per-Class Performance

| Class | Recall |
|---|---|
| No Tumor | 100% |
| Pituitary Tumor | 99.8% |
| Meningioma | 98.0% |
| Glioma | 83.5% |

Glioma recall is lower due to visual overlap with Meningioma on T1 non-contrast sequences. Contrast-enhanced MRI is recommended to confirm glioma findings.

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- An Anthropic API key (optional — template reports are used without one)
- The trained model file `brain_tumor_model.h5` (optional — demo mode runs without it)

### Installation

```bash
git clone https://github.com/your-username/neuroscan-ai.git
cd neuroscan-ai
pip install -r requirements.txt
```

### Running Locally

```bash
streamlit run streamlit_app.py
```

The app opens at `http://localhost:8501`.

### Configuration

**Anthropic API key** — create `.streamlit/secrets.toml`:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

**Model file from Google Drive** — set the environment variable before running:

```bash
export GDRIVE_FILE_ID="your_google_drive_file_id"
streamlit run streamlit_app.py
```

The app will download `brain_tumor_model.h5` automatically on first launch if the file is not present locally.

---

## Project Structure

```
neuroscan-ai/
├── streamlit_app.py              # Main application
├── brain_tumor_model.h5          # Trained model (not tracked in git — see below)
├── training_history.png          # Training curve plot (optional)
├── confusion_matrix_ensemble.png # Confusion matrix plot (optional)
├── samples/
│   ├── glioma.jpg
│   ├── meningioma.jpg
│   ├── pituitary.jpg
│   └── no_tumor.jpg
├── .streamlit/
│   └── secrets.toml              # API keys (not tracked in git)
├── docs/
│   └── screenshots/              # README screenshots go here
├── requirements.txt
└── README.md
```

> The model file `brain_tumor_model.h5` should be excluded from version control via `.gitignore` due to its size. Host it on Google Drive and reference the file ID via the `GDRIVE_FILE_ID` environment variable, or place it manually in the project root.

---

## Requirements

```
streamlit>=1.32.0
tensorflow>=2.13.0
numpy>=1.24.0
opencv-python-headless>=4.8.0
matplotlib>=3.7.0
Pillow>=10.0.0
anthropic>=0.25.0
gdown>=5.1.0
```

---

## Deploying to Streamlit Cloud

1. Push the repository to GitHub (without `brain_tumor_model.h5` and `secrets.toml`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo.
3. Under **Settings > Secrets**, add:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-..."
   ```
4. Under **Settings > Environment variables**, add:
   ```
   GDRIVE_FILE_ID = "your_file_id"
   ```
5. Set the branch to `main` (or `master`) and the main file path to `streamlit_app.py`.

The model will be downloaded from Google Drive on cold start.

---

## Dataset

The model was trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) by Masoud Nickparvar (Kaggle), which contains approximately 7,000 axial MRI images across four classes. T1 and T2 sequences are both represented; axial orientation is preferred for this classifier.

---

## How Grad-CAM Works

Grad-CAM (Gradient-weighted Class Activation Mapping) computes which spatial regions of an input image were most influential in the model's final prediction:

1. A sub-model is constructed that outputs both the final prediction logits and the feature maps of the last convolutional layer in the ResNet50V2 backbone.
2. A `tf.GradientTape` records the gradient of the predicted class score with respect to those feature maps.
3. Gradients are globally average-pooled to produce per-channel importance weights.
4. A weighted sum of the feature maps is computed and passed through ReLU, yielding a 7 × 7 spatial heatmap.
5. The heatmap is upscaled to the original image dimensions and overlaid in the jet colormap.

In demo mode (no model file), a synthetic heatmap is derived from MRI pixel intensity with a spatial Gaussian bias toward the central brain region. This is clearly labelled in the UI and is for demonstration purposes only.

---

## AI Clinical Report

When an Anthropic API key is configured, Claude Sonnet analyses the original MRI and the Grad-CAM overlay together and returns a structured JSON report with the following fields:

- Clinical interpretation
- Location and morphology
- Model reasoning
- Grad-CAM analysis
- Risk level (HIGH / MODERATE / LOW)
- Patient-friendly explanation
- Recommended next steps
- Image quality assessment
- Uncertainty factors
- Reliability score (0–100)

Without an API key, pre-written template reports are shown for each class.

---

## Disclaimer

This system is an AI-assisted decision-support tool built for research and educational purposes. It must not be used as a substitute for professional medical diagnosis, radiology review, or clinical judgment. All findings must be reviewed and confirmed by a licensed radiologist or neurosurgeon. The authors accept no liability for clinical decisions made on the basis of this system's output.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

- [Kaggle Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) — Masoud Nickparvar
- [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391) — Selvaraju et al., 2017
- [Anthropic Claude](https://anthropic.com) — AI report generation
- [Streamlit](https://streamlit.io) — Application framework
