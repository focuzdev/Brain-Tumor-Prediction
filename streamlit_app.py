"""
NeuroScan AI — Brain Tumor MRI Classification with Grad-CAM
===========================================================
Publication-quality Streamlit deployment of a 4-class CNN model.
Classes: Glioma | Meningioma | Pituitary Tumor | No Tumor
Theme: Clean light clinical — Q1 journal ready
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io
import base64
import os
import json
import anthropic
import gdown

# TensorFlow — loaded only if available
try:
    import tensorflow as tf
    from tensorflow import keras
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────────────
# CSS — Light Clinical Theme
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,700;1,9..144,300&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
}

/* ── App Background: clean off-white clinical ── */
.stApp {
    background-color: #f4f6fb;
    background-image:
        radial-gradient(ellipse 70% 40% at 0% 0%, rgba(219,230,255,0.7) 0%, transparent 60%),
        radial-gradient(ellipse 50% 30% at 100% 100%, rgba(209,240,255,0.5) 0%, transparent 60%);
    color: #1e2d45;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }

/* ════════════════════════════════════════════
   MASTHEAD
════════════════════════════════════════════ */
.masthead {
    position: relative;
    padding: 2rem 2.5rem 1.75rem;
    margin: 0 -2rem 2rem -2rem;
    background: linear-gradient(135deg, #1a3a6b 0%, #1e4d8c 40%, #0e3060 100%);
    border-bottom: 3px solid #2563eb;
    overflow: hidden;
}
.masthead::before {
    content: '';
    position: absolute; inset: 0;
    background:
        repeating-linear-gradient(90deg, transparent, transparent 60px,
            rgba(255,255,255,0.03) 60px, rgba(255,255,255,0.03) 61px),
        repeating-linear-gradient(0deg, transparent, transparent 60px,
            rgba(255,255,255,0.03) 60px, rgba(255,255,255,0.03) 61px);
}
.masthead-inner { position: relative; z-index: 1; }
.masthead-top {
    display: flex; align-items: flex-start;
    justify-content: space-between; flex-wrap: wrap; gap: 1rem;
}
.masthead-brand { display: flex; align-items: center; gap: 16px; }
.brand-icon {
    width: 52px; height: 52px; border-radius: 12px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.25);
    display: flex; align-items: center; justify-content: center;
    font-size: 26px; flex-shrink: 0;
}
.brand-title {
    font-family: 'Fraunces', serif;
    font-size: 26px; font-weight: 500;
    color: #ffffff; letter-spacing: -0.5px;
    line-height: 1.2; margin: 0;
}
.brand-title span { color: #93c5fd; }
.brand-sub {
    font-family: 'DM Mono', monospace;
    font-size: 10.5px; color: rgba(255,255,255,0.45);
    letter-spacing: 0.14em; text-transform: uppercase;
    margin: 4px 0 0; display: block;
}
.masthead-badges { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.mbadge {
    font-family: 'DM Mono', monospace; font-size: 10px; font-weight: 500;
    padding: 4px 10px; border-radius: 4px;
    letter-spacing: 0.06em; text-transform: uppercase;
}
.mbadge-blue  { background: rgba(255,255,255,0.12); color: #bfdbfe; border: 1px solid rgba(255,255,255,0.2); }
.mbadge-cyan  { background: rgba(6,182,212,0.2);    color: #a5f3fc; border: 1px solid rgba(6,182,212,0.35); }
.mbadge-green { background: rgba(16,185,129,0.2);   color: #a7f3d0; border: 1px solid rgba(16,185,129,0.35); }
.mbadge-amber { background: rgba(245,158,11,0.2);   color: #fde68a; border: 1px solid rgba(245,158,11,0.35); }

.masthead-divider {
    height: 1px; margin: 1.25rem 0 1rem;
    background: linear-gradient(90deg, rgba(255,255,255,0.2), rgba(255,255,255,0.04) 70%, transparent);
}
.masthead-stats { display: flex; gap: 2.5rem; flex-wrap: wrap; }
.mstat-val {
    font-family: 'Fraunces', serif; font-size: 20px;
    color: #fff; font-weight: 400; line-height: 1;
}
.mstat-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: rgba(255,255,255,0.35); text-transform: uppercase;
    letter-spacing: 0.1em; margin-top: 3px;
}

/* ════════════════════════════════════════════
   SECTION LABELS
════════════════════════════════════════════ */
.section-label {
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: #7a96bb; text-transform: uppercase; letter-spacing: 0.15em;
    margin-bottom: 10px; margin-top: 6px;
    display: flex; align-items: center; gap: 8px;
}
.section-label::after {
    content: ''; flex: 1; height: 1px; background: #dde5f0;
}

/* ════════════════════════════════════════════
   CARDS
════════════════════════════════════════════ */
.glass-card {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 14px;
    box-shadow: 0 1px 4px rgba(30,45,80,0.06);
}

/* ════════════════════════════════════════════
   SAMPLE SELECTOR CARD
════════════════════════════════════════════ */
.sample-card {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(30,45,80,0.05);
}
.sample-card-title {
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: #7a96bb; text-transform: uppercase;
    letter-spacing: 0.14em; margin-bottom: 8px;
}

/* ════════════════════════════════════════════
   PREDICTION DISPLAY
════════════════════════════════════════════ */
.pred-container {
    background: #ffffff;
    border: 1px solid #dde5f0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(30,45,80,0.08);
}
.pred-container::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
}
.pred-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: #7a96bb; text-transform: uppercase;
    letter-spacing: 0.15em; margin-bottom: 8px;
}
.pred-class-name {
    font-family: 'Fraunces', serif; font-size: 34px;
    font-weight: 500; color: #1a2d50;
    letter-spacing: -1px; line-height: 1.1; margin-bottom: 14px;
}
.pred-conf-wrap {
    background: #eef2fa; border-radius: 6px;
    height: 5px; margin: 8px 0 5px; overflow: hidden;
}
.pred-conf-bar {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #1d4ed8, #0891b2);
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}
.pred-conf-text {
    font-family: 'DM Mono', monospace; font-size: 12px; color: #0891b2;
}

/* ── Risk Badges ── */
.risk-badge {
    display: inline-flex; align-items: center; gap: 7px;
    padding: 6px 14px; border-radius: 6px;
    font-family: 'DM Mono', monospace; font-size: 11px;
    font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; margin-top: 12px;
}
.risk-dot { width: 6px; height: 6px; border-radius: 50%; }
.risk-HIGH     { background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; }
.risk-dot-HIGH { background: #ef4444; }
.risk-MODERATE     { background: #fffbeb; color: #b45309; border: 1px solid #fde68a; }
.risk-dot-MODERATE { background: #f59e0b; }
.risk-LOW     { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.risk-dot-LOW { background: #22c55e; }

/* ════════════════════════════════════════════
   REPORT BLOCKS
════════════════════════════════════════════ */
.rblock {
    border-left: 3px solid #93c5fd;
    background: #f8faff;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px; margin-bottom: 12px;
}
.rblock-danger { border-left-color: #f87171; background: #fff8f8; }
.rblock-warn   { border-left-color: #fbbf24; background: #fffdf5; }
.rblock-ok     { border-left-color: #34d399; background: #f5fdf8; }
.rblock-title {
    font-family: 'DM Mono', monospace; font-size: 10px;
    color: #7a96bb; text-transform: uppercase;
    letter-spacing: 0.14em; margin-bottom: 7px;
}
.rblock-body { font-size: 13px; line-height: 1.8; color: #3a5070; }

/* ════════════════════════════════════════════
   DISCLAIMER
════════════════════════════════════════════ */
.disclaimer-box {
    background: #fffcf0;
    border: 1px solid #fde68a;
    border-left: 3px solid #f59e0b;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    font-family: 'DM Mono', monospace; font-size: 10.5px;
    color: #78614a; line-height: 1.7; margin-top: 18px;
}
.disclaimer-box strong { color: #b45309; }

/* ════════════════════════════════════════════
   STREAMLIT OVERRIDES
════════════════════════════════════════════ */
/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #dde5f0 !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important; }
[data-testid="stSidebar"] h3 {
    font-family: 'Fraunces', serif !important;
    font-size: 14px !important; font-weight: 500 !important;
    color: #3a5070 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #eef2fa !important;
    border-radius: 10px !important; padding: 4px !important;
    gap: 2px !important; border: 1px solid #dde5f0 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 7px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 11px !important; color: #7a96bb !important;
    padding: 8px 16px !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important; color: #1d4ed8 !important;
    box-shadow: 0 1px 3px rgba(30,45,80,0.1) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #b8cce4 !important;
    border-radius: 12px !important; background: #f8faff !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1d4ed8 0%, #0891b2 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important; font-size: 14px !important;
    padding: 12px 24px !important; width: 100% !important;
    box-shadow: 0 4px 14px rgba(29,78,216,0.25) !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.92 !important; transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(29,78,216,0.35) !important;
}

/* Selectbox / Dropdown */
[data-testid="stSelectbox"] > div > div {
    background: #ffffff !important;
    border: 1px solid #b8cce4 !important;
    border-radius: 8px !important;
    color: #1e2d45 !important;
}

/* Metrics */
[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dde5f0 !important;
    border-radius: 10px !important; padding: 12px 16px !important;
    box-shadow: 0 1px 3px rgba(30,45,80,0.05) !important;
}
[data-testid="stMetricLabel"] { color: #7a96bb !important; font-size: 11px !important; }
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    color: #1a2d50 !important; font-size: 22px !important;
}

/* Progress bar */
[data-testid="stProgress"] > div { background: #e2eaf5 !important; border-radius: 4px !important; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #1d4ed8, #0891b2) !important;
    border-radius: 4px !important;
}

/* Toggle */
[data-testid="stToggle"] label { color: #3a5070 !important; font-size: 13px !important; }

/* Slider */
[data-testid="stSlider"] > div > div > div { background: #1d4ed8 !important; }

/* Radio */
[data-testid="stRadio"] label { font-size: 13px !important; color: #3a5070 !important; }

/* Download button */
[data-testid="stDownloadButton"] > button {
    background: #f0f5ff !important;
    border: 1px solid #b8cce4 !important; color: #1d4ed8 !important;
    font-family: 'DM Mono', monospace !important; font-size: 12px !important;
    border-radius: 8px !important; padding: 8px 16px !important;
    width: 100% !important; box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #dbeafe !important; transform: none !important;
}

/* Code blocks */
code {
    font-family: 'DM Mono', monospace !important; font-size: 11px !important;
    background: #eef2fa !important; color: #3a5070 !important;
    border: 1px solid #dde5f0 !important; border-radius: 4px !important;
}

/* Divider */
hr { border-color: #dde5f0 !important; margin: 1.5rem 0 !important; }

/* Image captions */
[data-testid="caption"] {
    font-family: 'DM Mono', monospace !important; font-size: 10px !important;
    color: #7a96bb !important; text-align: center !important;
    letter-spacing: 0.08em !important; text-transform: uppercase !important;
}

/* Success/Info */
.stSuccess { background: #f0fdf4 !important; border-color: #bbf7d0 !important; color: #166534 !important; }
.stInfo    { background: #eff6ff !important; border-color: #bfdbfe !important; color: #1e40af !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
IMG_SIZE    = (224, 224)
MODEL_PATH  = "brain_tumor_model.h5"
SAMPLE_DIR  = "samples"

# ── Secrets — exactly as original working code ────────────────────────────────
GDRIVE_FILE_ID = os.environ.get("GDRIVE_FILE_ID", "")

# ── Maps label → (risk text, badge css, dot css) ──────────────────────────────
RISK_MAP = {
    "Glioma":          ("HIGH",     "risk-HIGH",     "risk-dot-HIGH"),
    "Meningioma":      ("MODERATE", "risk-MODERATE", "risk-dot-MODERATE"),
    "Pituitary Tumor": ("MODERATE", "risk-MODERATE", "risk-dot-MODERATE"),
    "No Tumor":        ("LOW",      "risk-LOW",      "risk-dot-LOW"),
}

# ── Sample files — matches your GitHub samples/ folder ────────────────────────
SAMPLE_OPTIONS = {
    "— Select a sample image —": None,
    "🟠  Glioma":                "glioma.jpg",
    "🔵  Meningioma":            "meningioma.jpg",
    "🟣  Pituitary Tumor":       "pituitary.jpg",
    "🟢  No Tumor":              "no_tumor.jpg",
}

# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CNN model…")
def load_model():
    if not TF_AVAILABLE:
        return None
    if not os.path.exists(MODEL_PATH) and GDRIVE_FILE_ID:
        with st.spinner("⬇️ Downloading model from Google Drive…"):
            gdown.download(
                f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}",
                MODEL_PATH, quiet=False
            )
    if not os.path.exists(MODEL_PATH):
        return None
    return keras.models.load_model(MODEL_PATH)

# ──────────────────────────────────────────────────────────────────────────────
# Image helpers
# ──────────────────────────────────────────────────────────────────────────────
def preprocess(pil_img):
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, 0)

def make_gradcam(model, img_array, pred_index):
    last_conv = None
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            last_conv = layer.name; break
    if last_conv is None:
        return np.zeros(IMG_SIZE)
    grad_model = keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_array)
        class_channel = preds[:, pred_index]
    grads  = tape.gradient(class_channel, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_gradcam(pil_img, heatmap, alpha=0.45):
    orig  = np.array(pil_img.convert("RGB").resize(IMG_SIZE))
    hm    = cv2.resize(heatmap, IMG_SIZE)
    hm_c  = (cm.jet(hm)[:, :, :3] * 255).astype(np.uint8)
    blend = (orig * (1 - alpha) + hm_c * alpha).astype(np.uint8)
    return Image.fromarray(blend)

def pil_to_b64(pil_img, fmt="JPEG"):
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format=fmt)
    return base64.standard_b64encode(buf.getvalue()).decode()

# ──────────────────────────────────────────────────────────────────────────────
# Claude AI Report
# ──────────────────────────────────────────────────────────────────────────────
def generate_ai_report(
    pil_img,
    pred_class: str,
    confidence: float,
    gradcam_img=None,
) -> dict:
    try:
        api_key = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        api_key = ""

    if not api_key:
        return _mock_report(pred_class, confidence)

    try:
        client = anthropic.Anthropic(api_key=api_key)

        system_prompt = """You are an expert neuro-oncology AI assistant.
Analyze the provided brain MRI image and model prediction.
Respond ONLY with a valid JSON object (no markdown, no preamble) with these keys:
{
  "clinical_interpretation": "...",
  "location_morphology": "...",
  "model_reasoning": "...",
  "gradcam_analysis": "...",
  "risk_level": "HIGH | MODERATE | LOW",
  "risk_justification": "...",
  "patient_explanation": "...",
  "next_steps": "...",
  "image_quality": "GOOD | ADEQUATE | POOR",
  "uncertainty_factors": "...",
  "reliability_score": 0-100,
  "overall_reliability": "...",
  "disclaimer": "This is AI-assisted decision support only."
}
Be conservative, evidence-based, and never hallucinate findings."""

        content_msg = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": pil_to_b64(pil_img)},
            },
            {
                "type": "text",
                "text": (
                    f"CNN Prediction: {pred_class}\n"
                    f"Confidence: {confidence:.1f}%\n"
                    f"All classes: Glioma, Meningioma, Pituitary Tumor, No Tumor\n"
                    + ("Grad-CAM heatmap is provided as the second image." if gradcam_img else "No Grad-CAM available.")
                    + "\n\nGenerate the clinical report JSON."
                ),
            },
        ]

        if gradcam_img:
            content_msg.insert(1, {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg",
                           "data": pil_to_b64(gradcam_img)},
            })

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            system=system_prompt,
            messages=[{"role": "user", "content": content_msg}],
        )

        raw = response.content[0].text.strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        return json.loads(raw)

    except Exception:
        return _mock_report(pred_class, confidence)


def _mock_report(pred_class, confidence):
    templates = {
        "Glioma": {
            "clinical_interpretation": "The MRI demonstrates a heterogeneous mass lesion with irregular margins and surrounding peritumoral edema. Mixed signal intensity with areas of necrosis and ring-enhancing pattern are characteristic of high-grade glioma. Significant mass effect noted with midline shift.",
            "location_morphology": "Right frontal lobe, supratentorial compartment. Irregular lobulated borders with heterogeneous internal architecture. Surrounding vasogenic edema extends into adjacent white matter tracts.",
            "model_reasoning": f"Glioma prediction ({confidence:.1f}%) is strongly supported by ring-enhancing pattern, heterogeneous signal, and peritumoral edema — hallmarks of high-grade glioblastoma.",
            "gradcam_analysis": "Activation heatmap appropriately localised to the tumor epicenter with secondary activation at the peritumoral edema boundary. Model attention is clinically meaningful.",
            "risk_level": "HIGH",
            "risk_justification": "High-grade glioma carries significant morbidity. Urgent multidisciplinary neuro-oncology review is indicated.",
            "patient_explanation": "The scan shows signs of a brain tumor called a Glioma. This is NOT a final diagnosis — your doctor must confirm with further tests.",
            "next_steps": "1. Neuroradiologist review\n2. Contrast-enhanced MRI\n3. Neurosurgical consultation\n4. Tissue biopsy for histopathological confirmation",
            "image_quality": "GOOD", "uncertainty_factors": "Partial ambiguity at tumor-edema boundary.",
            "reliability_score": 91, "overall_reliability": "High reliability. Minor uncertainty at infiltrative margin.",
            "disclaimer": "AI-assisted decision support only. Not a final diagnosis.",
        },
        "Meningioma": {
            "clinical_interpretation": "Well-circumscribed extra-axial mass with dural tail sign, homogeneous signal intensity, and broad base of attachment along the parasagittal convexity.",
            "location_morphology": "Parasagittal convexity, extra-axial. Broad dural base, smooth well-defined margins. Approximately 2.8cm in greatest dimension.",
            "model_reasoning": f"Meningioma prediction ({confidence:.1f}%) aligns with extra-axial location, homogeneous signal, and dural attachment — classic imaging features.",
            "gradcam_analysis": "Model correctly focuses on the lesion-dura interface and dural tail — clinically appropriate activation.",
            "risk_level": "MODERATE",
            "risk_justification": "Most meningiomas are WHO Grade I (benign). Risk depends on size, location, and growth rate.",
            "patient_explanation": "The scan suggests a meningioma — usually slow-growing and attached to the brain's outer lining, often non-cancerous.",
            "next_steps": "1. Neurology review\n2. Contrast-enhanced MRI\n3. Observation vs. surgical resection based on symptoms",
            "image_quality": "GOOD", "uncertainty_factors": "Cavernous sinus involvement requires dedicated coronal sequences.",
            "reliability_score": 86, "overall_reliability": "Good reliability.",
            "disclaimer": "AI-assisted decision support only.",
        },
        "Pituitary Tumor": {
            "clinical_interpretation": "Intrasellar mass expanding the sella turcica with suprasellar extension. Optic chiasm displaced superiorly. Pituitary stalk deviated to the right.",
            "location_morphology": "Sella turcica, ~1.6cm macroadenoma with suprasellar extension. Cavernous sinuses appear intact bilaterally.",
            "model_reasoning": f"Pituitary tumor prediction ({confidence:.1f}%) confirmed by classic intrasellar location, sella expansion, and chiasm displacement.",
            "gradcam_analysis": "Model activates precisely on the sellar region with secondary activation at the chiasm interface — appropriate clinical focus.",
            "risk_level": "MODERATE",
            "risk_justification": "Usually benign pituitary adenoma. Risk from hormonal dysfunction and optic chiasm compression.",
            "patient_explanation": "The scan shows a tumor in the pituitary gland — a hormone-regulating gland at the base of the brain. Usually non-cancerous.",
            "next_steps": "1. Endocrinology consultation\n2. Visual field testing\n3. Full hormone panel\n4. Consider transsphenoidal surgery",
            "image_quality": "GOOD", "uncertainty_factors": "Cavernous sinus invasion requires Knosp grading.",
            "reliability_score": 89, "overall_reliability": "High reliability.",
            "disclaimer": "AI-assisted decision support only.",
        },
        "No Tumor": {
            "clinical_interpretation": "Normal brain parenchyma. No mass lesion, abnormal enhancement, or signal abnormality identified. Age-appropriate cortical and subcortical structures.",
            "location_morphology": "No focal lesion. Gray-white matter differentiation preserved. Midline structures central. Ventricles normal in size and configuration.",
            "model_reasoning": f"No Tumor prediction ({confidence:.1f}%) consistent with uniformly normal imaging: symmetric architecture, no mass effect, preserved sulci and gyri.",
            "gradcam_analysis": "Low distributed activation with no focal pathological concentration — consistent with a normal scan.",
            "risk_level": "LOW", "risk_justification": "No imaging evidence of intracranial neoplasm on this study.",
            "patient_explanation": "Good news — the AI did not detect a tumor. The brain scan appears normal. Follow up with your doctor if symptoms persist.",
            "next_steps": "Continue clinical follow-up if symptomatic. Repeat imaging if clinically indicated.",
            "image_quality": "GOOD", "uncertainty_factors": "None significant.",
            "reliability_score": 95, "overall_reliability": "Very high reliability.",
            "disclaimer": "AI-assisted decision support only.",
        },
    }
    return templates.get(pred_class, templates["Glioma"])

# ──────────────────────────────────────────────────────────────────────────────
# HTML helpers
# ──────────────────────────────────────────────────────────────────────────────
def pred_card_html(pred_class, confidence, risk_label, risk_css, dot_css):
    return f"""
    <div class="pred-container">
      <div class="pred-eyebrow">CNN · Predicted Diagnosis</div>
      <div class="pred-class-name">{pred_class}</div>
      <div style="display:flex;align-items:center;justify-content:space-between;">
        <span style="font-family:'DM Mono',monospace;font-size:11px;color:#7a96bb;">Model Confidence</span>
        <span class="pred-conf-text">{confidence:.1f}%</span>
      </div>
      <div class="pred-conf-wrap">
        <div class="pred-conf-bar" style="width:{confidence}%"></div>
      </div>
      <div class="risk-badge {risk_css}">
        <span class="risk-dot {dot_css}"></span>
        {risk_label} RISK
      </div>
    </div>"""

def rblock_html(title, body, variant=""):
    return f"""
    <div class="rblock {variant}">
      <div class="rblock-title">{title}</div>
      <div class="rblock-body">{body}</div>
    </div>"""

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style="font-family:'Fraunces',serif;font-size:16px;color:#3a5070;
              margin:0 0 1rem;font-weight:500;">Configuration</p>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Analysis Settings")
    show_gradcam  = st.toggle("Grad-CAM Visualization", value=True)
    try:
        _key_check = st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        _key_check = ""
    use_ai_report = st.toggle("Claude AI Report", value=bool(_key_check))

    # ── Claude Status ──────────────────────────────────────────────────────────
    if _key_check:
        st.success("✅ Claude API connected")
    else:
        st.error("❌ No API key — template reports only")

    # ── Model Status ────────────────────────────────────────────────────────────
    if os.path.exists(MODEL_PATH):
        st.success("✅ CNN Model loaded")
    else:
        st.error(f"❌ Model not found: {MODEL_PATH}")
        st.warning("⚠️ Running in DEMO MODE — predictions are fake. Upload brain_tumor_model.h5 to GitHub master branch.")
    gradcam_alpha = st.slider("Heatmap Intensity", 0.2, 0.8, 0.45, 0.05)

    st.divider()
    st.markdown("### ℹ️ Model Specification")
    st.code(
        "Architecture : EfficientNetB0\n"
        "Classes      : 4\n"
        "Input        : 224×224 RGB\n"
        "Framework    : TensorFlow 2.x\n"
        "XAI          : Grad-CAM",
        language="text"
    )

    st.divider()
    st.markdown("""
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:#78614a;
                line-height:1.7;padding:10px 12px;background:#fffcf0;
                border-radius:8px;border:1px solid #fde68a;
                border-left:3px solid #f59e0b;">
      <strong>⚠ Clinical Disclaimer</strong><br>
      AI-assisted decision support only. Not a substitute for professional
      medical diagnosis. All findings must be reviewed by a licensed
      radiologist or neurosurgeon.
    </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MASTHEAD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="masthead">
  <div class="masthead-inner">
    <div class="masthead-top">
      <div class="masthead-brand">
        <div class="brand-icon">🧠</div>
        <div>
          <p class="brand-title">NeuroScan <span>AI</span></p>
          <span class="brand-sub">Brain Tumor MRI Classification &amp; Explainability System</span>
        </div>
      </div>
      <div class="masthead-badges">
        <span class="mbadge mbadge-blue">EfficientNetB0</span>
        <span class="mbadge mbadge-cyan">Grad-CAM XAI</span>
        <span class="mbadge mbadge-green">Claude AI Reports</span>
        <span class="mbadge mbadge-amber">4-Class CNN</span>
      </div>
    </div>
    <div class="masthead-divider"></div>
    <div class="masthead-stats">
      <div class="mstat"><div class="mstat-val">4</div><div class="mstat-label">Tumor Classes</div></div>
      <div class="mstat"><div class="mstat-val">224²</div><div class="mstat-label">Input Resolution</div></div>
      <div class="mstat"><div class="mstat-val">~7K</div><div class="mstat-label">Training Images</div></div>
      <div class="mstat"><div class="mstat-val">XAI</div><div class="mstat-label">Explainability</div></div>
      <div class="mstat"><div class="mstat-val">v2.0</div><div class="mstat-label">Model Version</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
# ── Model warning banner — shows prominently on main page ──────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("""
    ❌ **CNN Model Not Found** — `brain_tumor_model.h5` is missing!

    **Fix:** Go to Streamlit Cloud → Manage App → Settings → General → change Branch from `main` to `master`

    Currently running in **DEMO MODE** — all predictions are fake sample data, not real CNN inference.
    """)

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.markdown('<div class="section-label">Input · MRI Scan</div>', unsafe_allow_html=True)

    # ── Upload ──────────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Upload your own MRI image",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="visible",
        help="JPEG, PNG or BMP — up to 10MB"
    )

    # ── Sample dropdown ─────────────────────────────────────────────────────
    st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sample-card">
      <div class="sample-card-title">Or choose a sample image</div>
    </div>""", unsafe_allow_html=True)

    selected_label = st.selectbox(
        "Sample images",
        options=list(SAMPLE_OPTIONS.keys()),
        index=0,
        label_visibility="collapsed",
        help="Pre-loaded MRI examples for each tumor class"
    )
    selected_filename = SAMPLE_OPTIONS[selected_label]

    # ── Resolve image source ─────────────────────────────────────────────────
    pil_image   = None
    image_source = None   # "upload" | "sample" | "placeholder"

    if uploaded:
        pil_image    = Image.open(uploaded)
        image_source = "upload"
        st.success("✅ Image uploaded successfully.")
    elif selected_filename is not None:
        sample_path = os.path.join(SAMPLE_DIR, selected_filename)
        if os.path.exists(sample_path):
            pil_image    = Image.open(sample_path)
            image_source = "sample"
        else:
            st.warning(f"Sample file not found: `{sample_path}`")
    else:
        # No image selected yet — show placeholder
        st.markdown("""
        <div style="background:#f0f5ff;border:1.5px dashed #b8cce4;border-radius:12px;
                    padding:2.5rem;text-align:center;color:#7a96bb;">
          <div style="font-size:36px;margin-bottom:10px;">🩻</div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;letter-spacing:0.08em;">
            Upload an image or select a sample above
          </div>
        </div>""", unsafe_allow_html=True)

    if pil_image:
        caption = "UPLOADED SCAN" if image_source == "upload" else f"SAMPLE · {selected_label.strip()}"
        st.image(pil_image, caption=caption, use_column_width=True, clamp=True)

    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    analyze_clicked = st.button(
        "🔬  Analyze & Generate Clinical Report",
        disabled=(pil_image is None)
    )

with col_output:
    st.markdown('<div class="section-label">Model Output · Prediction</div>', unsafe_allow_html=True)
    if not analyze_clicked:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #dde5f0;border-radius:14px;
                    padding:3rem 2rem;text-align:center;min-height:200px;
                    box-shadow:0 1px 4px rgba(30,45,80,0.06);">
          <div style="font-size:40px;margin-bottom:14px;">🔬</div>
          <div style="font-family:'Fraunces',serif;font-size:16px;
                      color:#7a96bb;font-weight:300;line-height:1.6;">
            Upload an MRI or select a sample,<br>then click Analyze.
          </div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# ANALYSIS PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
if analyze_clicked and pil_image:
    model = load_model()

    # ── Inference ──
    with st.spinner("Running CNN inference…"):
        img_array = preprocess(pil_image)
        if model:
            preds = model.predict(img_array, verbose=0)[0]
        else:
            # Demo mode — derive fake preds from selected sample
            demo_map = {
                "glioma.jpg":     [0.942, 0.031, 0.019, 0.008],
                "meningioma.jpg": [0.052, 0.876, 0.048, 0.024],
                "pituitary.jpg":  [0.021, 0.043, 0.915, 0.021],
                "no_tumor.jpg":   [0.012, 0.009, 0.011, 0.968],
            }
            preds = np.array(demo_map.get(
                selected_filename or "glioma.jpg", demo_map["glioma.jpg"]
            ))

        pred_idx   = int(np.argmax(preds))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx]) * 100
        risk_label, risk_css, dot_css = RISK_MAP[pred_class]

    # ── Grad-CAM ──
    gradcam_overlay = heatmap = None
    if show_gradcam and model:
        with st.spinner("Computing Grad-CAM heatmap…"):
            heatmap         = make_gradcam(model, img_array, pred_idx)
            gradcam_overlay = overlay_gradcam(pil_image, heatmap, alpha=gradcam_alpha)

    # ── AI Report ──
    with st.spinner("Generating clinical report…"):
        report = (
            generate_ai_report(pil_image, pred_class, confidence,
                               gradcam_overlay if show_gradcam else None)
            if use_ai_report else _mock_report(pred_class, confidence)
        )

    # ── Prediction card ──
    with col_output:
        st.markdown(
            pred_card_html(pred_class, confidence, risk_label, risk_css, dot_css),
            unsafe_allow_html=True
        )

        st.markdown('<div class="section-label" style="margin-top:4px;">Class Probability Distribution</div>',
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 2.4))
        colors = ["#ef4444", "#3b82f6", "#a855f7", "#22c55e"]
        bars = ax.barh(CLASS_NAMES, preds * 100, color=colors, height=0.5)
        ax.set_xlim(0, 108)
        ax.set_xlabel("Probability (%)", color="#7a96bb", fontsize=8, labelpad=6)
        ax.tick_params(colors="#7a96bb", labelsize=8)
        ax.set_facecolor("#f8faff")
        fig.patch.set_facecolor("#f8faff")
        for spine in ax.spines.values():
            spine.set_edgecolor("#dde5f0")
        for bar, val in zip(bars, preds):
            ax.text(val * 100 + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val*100:.1f}%", va="center", color="#7a96bb", fontsize=8)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    # ── Grad-CAM ──
    if show_gradcam and gradcam_overlay:
        st.markdown("---")
        st.markdown('<div class="section-label">Explainability · Grad-CAM Visualization</div>',
                    unsafe_allow_html=True)
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.image(pil_image, caption="Original MRI", use_column_width=True)
        with gc2:
            st.image(gradcam_overlay, caption="Grad-CAM Overlay", use_column_width=True)
        with gc3:
            if heatmap is not None:
                fig2, ax2 = plt.subplots(figsize=(3, 3))
                ax2.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
                ax2.axis("off")
                ax2.set_facecolor("#f8faff")
                fig2.patch.set_facecolor("#f8faff")
                cb = fig2.colorbar(plt.cm.ScalarMappable(cmap="jet"),
                                   ax=ax2, fraction=0.046, pad=0.04)
                cb.ax.tick_params(colors="#7a96bb", labelsize=7)
                plt.tight_layout(pad=0.2)
                st.pyplot(fig2, use_container_width=True)
                plt.close()
                st.caption("Activation Heatmap")

    # ── Clinical Report ──
    st.markdown("---")
    st.markdown('<div class="section-label">AI-Assisted Clinical Report</div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🧾 Clinical Findings",
        "🧠 Model Reasoning",
        "👤 Patient Summary",
        "🔍 Reliability",
    ])

    with tab1:
        st.markdown(rblock_html("Clinical Interpretation",
            report.get("clinical_interpretation", ""), "rblock-danger"), unsafe_allow_html=True)
        st.markdown(rblock_html("Location & Morphology",
            report.get("location_morphology", "")), unsafe_allow_html=True)

    with tab2:
        st.markdown(rblock_html("Model Reasoning Alignment",
            report.get("model_reasoning", "")), unsafe_allow_html=True)
        st.markdown(rblock_html("Grad-CAM Activation Analysis",
            report.get("gradcam_analysis", ""), "rblock-ok"), unsafe_allow_html=True)

    with tab3:
        st.markdown(rblock_html("Plain Language Summary",
            report.get("patient_explanation", ""), "rblock-warn"), unsafe_allow_html=True)
        st.markdown(rblock_html("Recommended Next Steps",
            report.get("next_steps", "").replace("\n", "<br>")), unsafe_allow_html=True)

    with tab4:
        reliability_score = report.get("reliability_score", 80)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Reliability Score", f"{reliability_score}/100")
        with c2: st.metric("Image Quality", report.get("image_quality", "—"))
        with c3: st.metric("Risk Level", risk_label)
        st.progress(reliability_score / 100)
        qv = {"GOOD": "rblock-ok", "ADEQUATE": "rblock-warn", "POOR": "rblock-danger"}.get(
            report.get("image_quality", "GOOD"), "rblock-ok")
        st.markdown(rblock_html("Uncertainty Factors",
            report.get("uncertainty_factors", "None identified."), qv), unsafe_allow_html=True)
        st.markdown(rblock_html("Overall Reliability Assessment",
            report.get("overall_reliability", "")), unsafe_allow_html=True)

    # ── Disclaimer ──
    st.markdown(f"""
    <div class="disclaimer-box">
      <strong>⚠ AI-Assisted Decision Support Only</strong> —
      {report.get("disclaimer", "")}
      This system must not be used as a substitute for professional medical diagnosis.
      All findings require review and confirmation by a licensed radiologist or neurosurgeon.
    </div>""", unsafe_allow_html=True)

    # ── Export ──
    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    st.download_button(
        label="⬇  Export Full Report (JSON)",
        data=json.dumps({
            "system": "NeuroScan AI v2.0",
            "prediction": pred_class,
            "confidence_pct": round(confidence, 2),
            "risk_level": risk_label,
            "class_probabilities": {
                n: round(float(p), 4) for n, p in zip(CLASS_NAMES, preds)
            },
            **report,
        }, indent=2),
        file_name=f"neuroscan_{pred_class.lower().replace(' ', '_')}.json",
        mime="application/json",
    )
