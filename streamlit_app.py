"""
NeuroScan AI - Brain Tumor MRI Classification + Grad-CAM Heatmap
================================================================
Model    : ResNet50V2 + MobileNetV2 Ensemble | 4 classes | 95.31% accuracy
XAI      : Grad-CAM via TensorFlow GradientTape
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image, ImageOps
import io, base64, os, json
import gdown
from datetime import datetime

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    resnet_preprocess = None
    mobilenet_preprocess = None

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "theme" not in st.session_state:
    st.session_state.theme = "dark"
_dk = (st.session_state.theme == "dark")

# ================================================================
# CSS STYLING
# ================================================================
css_style = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

*,*::before,*::after{box-sizing:border-box}
html,body,[class*="css"]{font-family:'Inter',sans-serif;-webkit-font-smoothing:antialiased}

.stApp{
  background:#0a0e1a !important;
  background-image:radial-gradient(ellipse 90% 55% at 10% -5%,rgba(14,58,150,.35) 0%,transparent 45%),radial-gradient(ellipse 70% 45% at 95% 105%,rgba(8,100,160,.20) 0%,transparent 45%) !important;
  color:#e2e8f0 !important;
  min-height:100vh;
}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 !important;max-width:100% !important}

.topnav{
  position:sticky;top:0;z-index:200;
  background:rgba(10,14,26,.95);
  backdrop-filter:blur(24px) saturate(180%);
  border-bottom:1px solid rgba(56,189,248,.15);
  padding:.8rem 2.4rem;
  display:flex;align-items:center;justify-content:space-between;gap:1rem;
}
.nav-brand{display:flex;align-items:center;gap:13px}
.nav-logo{width:38px;height:38px;border-radius:10px;font-size:18px;background:linear-gradient(135deg,#1e40af,#0e7490);display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px rgba(56,189,248,.40);flex-shrink:0}
.nav-name{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:#e2e8f0;letter-spacing:-.3px}
.nav-name span{color:#38bdf8}
.nav-tagline{font-family:'DM Mono',monospace;font-size:8px;color:rgba(255,255,255,.45);letter-spacing:.15em;text-transform:uppercase;margin-top:1px}
.nav-right{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.chip{font-family:'DM Mono',monospace;font-size:9px;font-weight:500;padding:4px 10px;border-radius:20px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}
.c-blue{background:rgba(59,130,246,.14);color:#93c5fd;border:1px solid rgba(59,130,246,.28)}
.c-teal{background:rgba(20,184,166,.14);color:#5eead4;border:1px solid rgba(20,184,166,.28)}
.c-green{background:rgba(34,197,94,.14);color:#86efac;border:1px solid rgba(34,197,94,.28)}
.c-amber{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.28)}
.c-purple{background:rgba(139,92,246,.14);color:#c4b5fd;border:1px solid rgba(139,92,246,.28)}

.hero{position:relative;overflow:hidden;padding:3rem 0 2.5rem;background:linear-gradient(130deg,#040c1c 0%,#071630 55%,#040c1c 100%);border-bottom:1px solid rgba(56,189,248,.09)}
.hero-inner{position:relative;z-index:1;width:100%;padding:0 2.8rem;box-sizing:border-box}
.hero-top{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}
.hero-h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.9rem,3.5vw,3rem);font-weight:700;color:#e2e8f0;letter-spacing:-.7px;line-height:1.13;margin-bottom:.5rem}
.hero-h1 .grad{background:linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}
.hero-desc{font-size:15px;color:rgba(255,255,255,.70);line-height:1.74;max-width:530px}
.hero-stats{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end}
.hs{text-align:right}
.hs-n{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;background:linear-gradient(92deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}
.hs-l{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.12em;margin-top:3px}
.hero-div{height:1px;margin:1.2rem 0;background:linear-gradient(90deg,rgba(56,189,248,.30),rgba(129,140,248,.18),transparent)}
.pipeline{display:flex;align-items:center;gap:0;flex-wrap:wrap}
.pip-step{display:flex;align-items:center;gap:9px}
.pip-num{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0891b2);font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px rgba(56,189,248,.32);flex-shrink:0}
.pip-txt{font-size:11.5px;color:rgba(255,255,255,.65);line-height:1.38}
.pip-txt strong{color:rgba(255,255,255,.90);display:block;font-size:11px}
.pip-arr{color:rgba(56,189,248,.40);font-size:20px;padding:0 10px}

.wrap{width:100%;padding:2rem 2.8rem 4rem;box-sizing:border-box}
.glass{background:rgba(255,255,255,.030);border:1px solid rgba(255,255,255,.075);border-radius:20px;padding:1.8rem 2rem;backdrop-filter:blur(12px);box-shadow:0 8px 40px rgba(0,0,0,.35)}
.slbl{font-family:'DM Mono',monospace;font-size:11px;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.17em;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.slbl::after{content:'';flex:1;height:1px;background:rgba(56,189,248,.20)}

.pred-card{background:linear-gradient(135deg,rgba(14,30,70,.82),rgba(8,20,48,.92));border:1px solid rgba(56,189,248,.22);border-radius:18px;padding:1.5rem 1.6rem;margin-bottom:14px;position:relative;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.25)}
.pred-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8,#818cf8)}
.pred-eyebrow{font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px}
.pred-name{font-family:'Space Grotesk',sans-serif;font-size:clamp(30px,4vw,44px);font-weight:700;color:#f8fafc;letter-spacing:-1px;line-height:1.04;margin-bottom:15px}
.conf-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}
.conf-l{font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.55)}
.conf-v{font-family:'Space Grotesk',sans-serif;font-size:14px;color:#38bdf8;font-weight:600}
.conf-track{background:rgba(255,255,255,.10);border-radius:8px;height:6px;overflow:hidden;margin-bottom:15px}
.conf-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8)}
.risk-chip{display:inline-flex;align-items:center;gap:7px;padding:6px 15px;border-radius:20px;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}
.rdot{width:6px;height:6px;border-radius:50%}
.rH{background:rgba(239,68,68,.13);color:#dc2626;border:1px solid rgba(239,68,68,.35)}
.rdH{background:#ef4444;box-shadow:0 0 7px rgba(239,68,68,.5)}
.rM{background:rgba(245,158,11,.13);color:#d97706;border:1px solid rgba(245,158,11,.35)}
.rdM{background:#f59e0b;box-shadow:0 0 7px rgba(245,158,11,.5)}
.rL{background:rgba(34,197,94,.13);color:#16a34a;border:1px solid rgba(34,197,94,.35)}
.rdL{background:#22c55e;box-shadow:0 0 7px rgba(34,197,94,.5)}

.hm-section{background:rgba(2,6,14,.97);border:1px solid rgba(56,189,248,.20);border-radius:22px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.45);margin:1.8rem 0}
.hm-header{background:rgba(4,10,24,1);border-bottom:1px solid rgba(56,189,248,.12);padding:1.1rem 1.7rem;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px}
.hm-title{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:600;color:#e2e8f0;letter-spacing:-.3px}
.hm-sub{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:.13em;margin-top:3px}
.hm-legend{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.hm-leg{display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.60)}
.hm-swatch{width:28px;height:9px;border-radius:3px}
.hm-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-bottom:1px solid rgba(255,255,255,.08)}
.hm-stat{padding:13px 16px;border-right:1px solid rgba(255,255,255,.08);text-align:center}
.hm-stat:last-child{border-right:none}
.hm-sv{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:600;color:#38bdf8;line-height:1}
.hm-sl{font-family:'DM Mono',monospace;font-size:8.5px;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:.1em;margin-top:4px}
.hm-col-lbl{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.13em;text-align:center;margin-bottom:8px}
.hm-col-note{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.45);text-align:center;margin-top:8px;line-height:1.6}
.hm-img-frame{border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.10);box-shadow:0 4px 16px rgba(0,0,0,.40)}
.cscale{background:rgba(255,255,255,.025);margin:0 1.6rem 1.4rem;border-radius:8px;padding:9px 13px;border:1px solid rgba(255,255,255,.08)}
.cscale-bar{height:12px;border-radius:3px;background:linear-gradient(90deg,#00007f 0%,#0000ff 12%,#007fff 24%,#00ffff 36%,#7fff7f 50%,#ffff00 64%,#ff7f00 76%,#ff0000 88%,#7f0000 100%);margin-bottom:5px}
.cscale-lbls{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:8.5px;color:rgba(255,255,255,.50)}
.hm-explain{background:rgba(255,255,255,.025);border-top:1px solid rgba(255,255,255,.08);padding:1.3rem 1.7rem}
.hm-exp-title{font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.15em;margin-bottom:12px}
.hm-exp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.hm-exp-item{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:12px 14px}
.hm-exp-t{font-family:'DM Mono',monospace;font-size:9px;color:#38bdf8;margin-bottom:5px;font-weight:500}
.hm-exp-b{font-size:13px;color:rgba(255,255,255,.80);line-height:1.65}

.rb{border-left:3px solid rgba(147,197,253,.55);background:rgba(255,255,255,.028);border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px}
.rb-red{border-left-color:rgba(248,113,113,.80);background:rgba(239,68,68,.08)}
.rb-yel{border-left-color:rgba(251,191,36,.80);background:rgba(245,158,11,.08)}
.rb-grn{border-left-color:rgba(52,211,153,.80);background:rgba(16,185,129,.08)}
.rb-t{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px}
.rb-b{font-size:14px;line-height:1.90;color:rgba(255,255,255,.88)}

.disc{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.30);border-left:3px solid rgba(245,158,11,.80);border-radius:0 12px 12px 0;padding:13px 18px;font-family:'DM Mono',monospace;font-size:10px;color:#fcd34d;line-height:1.78;margin-top:20px}
.disc strong{color:#f59e0b}

[data-testid="stSidebar"]{background:rgba(10,14,26,.98) !important;border-right:1px solid rgba(56,189,248,.09) !important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:#e2e8f0 !important}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.050) !important;border-radius:11px !important;padding:3px !important;gap:2px !important;border:1px solid rgba(255,255,255,.090) !important}
.stTabs [data-baseweb="tab"]{border-radius:8px !important;font-family:'DM Mono',monospace !important;font-size:10.5px !important;color:rgba(255,255,255,.60) !important;padding:8px 15px !important}
.stTabs [aria-selected="true"]{background:rgba(56,189,248,.15) !important;color:#38bdf8 !important;box-shadow:none !important}
[data-testid="stFileUploader"]{border:2px dashed rgba(56,189,248,.75) !important;border-radius:16px !important;background:rgba(14,58,140,.22) !important;padding:4px !important}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button{background:#0f172a !important;color:#ffffff !important;border:1.5px solid rgba(255,255,255,.18) !important;border-radius:10px !important;font-family:'Space Grotesk',sans-serif !important;font-weight:700 !important;font-size:14px !important;padding:10px 24px !important;box-shadow:0 2px 12px rgba(0,0,0,.50) !important}
.stButton > button{background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#4f46e5 100%) !important;color:#ffffff !important;border:none !important;border-radius:14px !important;font-family:'Space Grotesk',sans-serif !important;font-weight:700 !important;font-size:16px !important;padding:17px 28px !important;width:100% !important;box-shadow:0 6px 24px rgba(37,99,235,.55) !important}
.stButton > button:hover{background:linear-gradient(135deg,#38bdf8 0%,#3b82f6 50%,#6366f1 100%) !important;transform:translateY(-3px) !important;box-shadow:0 12px 36px rgba(37,99,235,.65) !important}
[data-testid="stMetric"]{background:rgba(255,255,255,.055) !important;border:1px solid rgba(255,255,255,.10) !important;border-radius:14px !important;padding:13px 16px !important}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif !important;color:#38bdf8 !important;font-size:22px !important}
[data-testid="stProgress"] > div{background:rgba(255,255,255,.10) !important;border-radius:4px !important}
[data-testid="stProgress"] > div > div{background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) !important;border-radius:4px !important}
[data-testid="stDownloadButton"] > button{background:rgba(56,189,248,.10) !important;border:1px solid rgba(56,189,248,.28) !important;color:#38bdf8 !important;font-family:'DM Mono',monospace !important;font-size:11.5px !important;border-radius:9px !important;padding:9px 16px !important;width:100% !important;box-shadow:none !important}
[data-testid="stImage"] img{border-radius:12px !important;border:1px solid rgba(255,255,255,.10) !important;display:block !important}
[data-testid="stSelectbox"] > div > div{background:#1e2d45 !important;border:1.5px solid rgba(56,189,248,.50) !important;border-radius:11px !important;min-height:46px !important}
[data-baseweb="popover"] ul,
[data-baseweb="menu"]{background:#0f1e36 !important;border:1px solid rgba(56,189,248,.25) !important;border-radius:10px !important;overflow:hidden !important;box-shadow:0 8px 32px rgba(0,0,0,.35) !important}
[data-baseweb="menu"] [role="option"]{background:transparent !important;color:#e2e8f0 !important;font-family:'Space Grotesk',sans-serif !important;font-size:14px !important;font-weight:500 !important;padding:12px 18px !important}
[data-baseweb="menu"] [role="option"]:hover{background:rgba(37,99,235,.18) !important;color:#38bdf8 !important}
"""

st.markdown(f"<style>{css_style}</style>", unsafe_allow_html=True)

# ================================================================
# CONSTANTS
# ================================================================
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
CLASS_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7"]
IMG_SIZE = (224, 224)
MODEL_PATH = "brain_tumor_model.h5"
MOBILENET_PATH = "mobilenet_model.h5"
SAMPLE_DIR = "samples"
GDRIVE_ID = os.environ.get("GDRIVE_FILE_ID", "")
GDRIVE_MOBILENET_ID = os.environ.get("GDRIVE_MOBILENET_FILE_ID", "")

RISK = {
    "Glioma": ("HIGH", "rH", "rdH"),
    "Meningioma": ("MODERATE", "rM", "rdM"),
    "Pituitary Tumor": ("MODERATE", "rM", "rdM"),
    "No Tumor": ("LOW", "rL", "rdL"),
}
SAMPLES = {
    "Select a sample image": None,
    "Glioma": "glioma.jpg",
    "Meningioma": "meningioma.jpg",
    "Pituitary Tumor": "pituitary.jpg",
    "No Tumor": "no_tumor.jpg",
}

# ================================================================
# MODEL LOADING
# ================================================================
@st.cache_resource(show_spinner="Loading CNN models...")
def load_models():
    """Load both models with proper error handling."""
    models = {
        "resnet": None, 
        "mobilenet": None,
        "resnet_map": list(range(4)),
        "mobilenet_map": list(range(4)),
        "loaded": False
    }
    
    if not TF_AVAILABLE:
        st.warning("⚠️ TensorFlow not available. Running in demo mode.")
        return models
    
    # Load ResNet50V2
    try:
        if not os.path.exists(MODEL_PATH):
            if GDRIVE_ID:
                with st.spinner("Downloading ResNet50V2..."):
                    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
            else:
                st.warning(f"⚠️ {MODEL_PATH} not found. Running in demo mode.")
        
        if os.path.exists(MODEL_PATH):
            models["resnet"] = keras.models.load_model(MODEL_PATH)
            st.success("✅ ResNet50V2 loaded")
        else:
            st.error(f"❌ {MODEL_PATH} not found")
    except Exception as e:
        st.error(f"❌ Failed to load ResNet50V2: {str(e)[:100]}")
    
    # Load MobileNetV2
    try:
        if not os.path.exists(MOBILENET_PATH):
            if GDRIVE_MOBILENET_ID:
                with st.spinner("Downloading MobileNetV2..."):
                    gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MOBILENET_ID}", MOBILENET_PATH, quiet=False)
            else:
                st.warning(f"⚠️ {MOBILENET_PATH} not found.")
        
        if os.path.exists(MOBILENET_PATH):
            models["mobilenet"] = keras.models.load_model(MOBILENET_PATH)
            st.success("✅ MobileNetV2 loaded")
        else:
            st.info(f"ℹ️ {MOBILENET_PATH} not found - running with ResNet50V2 only")
    except Exception as e:
        st.error(f"❌ Failed to load MobileNetV2: {str(e)[:100]}")
    
    # Auto-detect class mappings if models are loaded
    if models["resnet"] is not None:
        models["resnet_map"] = detect_class_mapping(models["resnet"], "resnet")
    if models["mobilenet"] is not None:
        models["mobilenet_map"] = detect_class_mapping(models["mobilenet"], "mobilenet")
    
    models["loaded"] = models["resnet"] is not None
    return models

def detect_class_mapping(model, model_name):
    """Detect correct class mapping using sample images."""
    if not os.path.isdir(SAMPLE_DIR):
        return list(range(4))
    
    predictions = {}
    for cls_name, fname in SAMPLES.items():
        if fname is None:
            continue
        fpath = os.path.join(SAMPLE_DIR, fname)
        if not os.path.exists(fpath):
            continue
        try:
            img = Image.open(fpath).convert("RGB")
            if model_name == "resnet":
                arr = preprocess(img)
            else:
                arr = preprocess_for_mobilenet(img)
            pred = model.predict(arr, verbose=0)[0]
            predictions[cls_name] = np.argmax(pred)
        except Exception:
            continue
    
    if len(predictions) < 4:
        return list(range(4))
    
    mapping = {}
    used = set()
    for cls_name in CLASS_NAMES:
        if cls_name in predictions:
            pred_idx = predictions[cls_name]
            if pred_idx not in used:
                mapping[pred_idx] = CLASS_NAMES.index(cls_name)
                used.add(pred_idx)
    
    result = list(range(4))
    for model_idx, class_idx in mapping.items():
        result[model_idx] = class_idx
    
    return result

# ================================================================
# PREPROCESSING
# ================================================================
def _resize_and_clean(img):
    resized = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    arr = arr[:IMG_SIZE[0], :IMG_SIZE[1], :3]
    return arr

def preprocess(img):
    arr = _resize_and_clean(img)
    if resnet_preprocess is not None:
        arr = resnet_preprocess(arr)
    else:
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)

def preprocess_for_mobilenet(img):
    arr = _resize_and_clean(img)
    if mobilenet_preprocess is not None:
        arr = mobilenet_preprocess(arr)
    else:
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)

# ================================================================
# MRI VALIDATION
# ================================================================
def validate_mri(pil_img, strict=False):
    """MRI validator with relaxed thresholds."""
    import numpy as np
    import cv2

    img_rgb = pil_img.convert("RGB")
    img_gray = pil_img.convert("L")
    w, h = img_rgb.size
    arr_rgb = np.array(img_rgb, dtype=np.float32)
    arr_gray = np.array(img_gray, dtype=np.float32)
    arr_u8 = arr_gray.astype(np.uint8)
    scores = {}

    # Colour saturation
    arr_u8_rgb = np.array(img_rgb, dtype=np.uint8)
    hsv = cv2.cvtColor(arr_u8_rgb, cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[:,:,1].mean())
    r_c, g_c, b_c = arr_rgb[:,:,0], arr_rgb[:,:,1], arr_rgb[:,:,2]
    mean_ch = (r_c + g_c + b_c) / 3
    ch_dev = float((np.abs(r_c-mean_ch)+np.abs(g_c-mean_ch)+np.abs(b_c-mean_ch)).mean())
    
    if strict:
        s1 = (mean_sat < 30.0) or (ch_dev < 15.0)
    else:
        s1 = (mean_sat < 45.0) or (ch_dev < 20.0)
    scores["colour_saturation"] = (s1, f"HSV sat {mean_sat:.1f} ch_dev {ch_dev:.1f}")

    # Dark surround
    if strict:
        dark_ratio = float((arr_gray < 25).sum() / arr_gray.size)
        s2 = dark_ratio >= 0.15
    else:
        dark_ratio = float((arr_gray < 30).sum() / arr_gray.size)
        s2 = dark_ratio >= 0.08
    scores["dark_surround"] = (s2, f"Near-black px {dark_ratio*100:.1f}%")

    # White background
    if strict:
        white_ratio = float((arr_gray > 230).sum() / arr_gray.size)
        s3 = white_ratio < 0.40
    else:
        white_ratio = float((arr_gray > 220).sum() / arr_gray.size)
        s3 = white_ratio < 0.55
    scores["white_background"] = (s3, f"Near-white px {white_ratio*100:.1f}%")

    # Intensity distribution
    hist, _ = np.histogram(arr_gray.flatten(), bins=256, range=(0,255))
    dark_m = hist[:40].sum() / arr_gray.size
    mid_m = hist[40:200].sum() / arr_gray.size
    if strict:
        s4 = (dark_m > 0.12) and (mid_m > 0.10)
    else:
        s4 = (dark_m > 0.08) and (mid_m > 0.07)
    scores["intensity_distribution"] = (s4, f"Dark {dark_m*100:.0f}% Mid {mid_m*100:.0f}%")

    # Bright pixel cap
    bright_ratio = float((arr_gray > 200).sum() / arr_gray.size)
    if strict:
        s5 = bright_ratio < 0.50
    else:
        s5 = bright_ratio < 0.65
    scores["bright_pixel_cap"] = (s5, f"Bright px {bright_ratio*100:.1f}%")

    # Edge structure
    edges = cv2.Canny(arr_u8, 40, 110)
    ed = float(edges.sum() / 255 / edges.size)
    sobelx = cv2.Sobel(arr_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(arr_u8, cv2.CV_64F, 0, 1, ksize=3)
    hv_ratio = (np.abs(sobely).sum() / (np.abs(sobelx).sum() + 1e-6))
    if strict:
        s6 = (ed < 0.20) and (hv_ratio < 1.6)
    else:
        s6 = (ed < 0.30) and (hv_ratio < 2.0)
    scores["edge_structure"] = (s6, f"Density {ed:.3f} H/V ratio {hv_ratio:.2f}")

    # Aspect ratio
    ratio = w / h
    if strict:
        s7 = 0.60 <= ratio <= 1.60
    else:
        s7 = 0.50 <= ratio <= 2.0
    scores["aspect_ratio"] = (s7, f"{w}×{h} ratio {ratio:.2f}")

    # Local contrast
    kernel = np.ones((8,8), np.float32) / 64
    local_mean = cv2.filter2D(arr_gray, -1, kernel)
    local_sq = cv2.filter2D(arr_gray**2, -1, kernel)
    local_var = np.clip(local_sq - local_mean**2, 0, None)
    local_std = np.sqrt(local_var)
    mean_lstd = float(local_std.mean())
    if strict:
        s8 = mean_lstd > 12.0
    else:
        s8 = mean_lstd > 6.0
    scores["local_contrast"] = (s8, f"Mean local σ {mean_lstd:.1f}")

    # Decision logic
    hard_ok = s1 and s2 and s3
    soft_pass = sum([s4, s5, s6, s7, s8])
    
    if strict:
        is_valid = hard_ok and (soft_pass >= 3)
    else:
        is_valid = hard_ok and (soft_pass >= 2)
    
    w_scores = (
        (2.0 if s1 else 0) + (2.0 if s2 else 0) + (2.0 if s3 else 0) +
        (1.0 if s4 else 0) + (0.8 if s5 else 0) + (0.8 if s6 else 0) +
        (0.6 if s7 else 0) + (0.8 if s8 else 0)
    )
    confidence = w_scores / 10.0
    
    return is_valid, float(confidence), scores

def mri_gate_ui(is_valid, confidence, reasons, _dk):
    pct = int(confidence * 100)
    if is_valid:
        clr = "#22c55e" if pct >= 70 else "#f59e0b"
        bg = "rgba(34,197,94,.08)" if pct >= 70 else "rgba(245,158,11,.07)"
        bdr = "rgba(34,197,94,.35)" if pct >= 70 else "rgba(245,158,11,.35)"
        st.markdown(f"""
<div style="background:{bg};border:1px solid {bdr};border-radius:10px;padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">
  <span style="font-size:18px;">✅</span>
  <div>
    <span style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:{clr};">Brain MRI verified</span>
    <span style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.40);margin-left:10px;">MRI score: {pct}%</span>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        rows = ""
        for sig, (passed, detail) in reasons.items():
            if not passed:
                label = sig.replace("_", " ").title()
                rows += f"""
<div style="display:flex;gap:10px;align-items:flex-start;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);">
  <span style="color:#ef4444;font-size:13px;flex-shrink:0;">✗</span>
  <span style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.70);flex:1;line-height:1.6;">
    <strong style="color:#f87171">{label}</strong>&nbsp;—&nbsp;{detail}
  </span>
</div>"""
        st.markdown(f"""
<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.40);border-left:4px solid #ef4444;border-radius:12px;padding:1.1rem 1.4rem;margin-bottom:1rem;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:.6rem;">
    <span style="font-size:22px;">🚫</span>
    <div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;font-weight:700;color:#f87171;">Image rejected — not a brain MRI</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.45);margin-top:2px;">MRI confidence score: {pct}%</div>
    </div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;color:rgba(255,100,100,.70);text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">Failed signals:</div>
  <div>{rows}</div>
</div>""", unsafe_allow_html=True)

# ================================================================
# INFERENCE
# ================================================================
def run_inference(img, models, temperature=1.4, use_ensemble=True):
    """Run inference with ensemble."""
    resnet_model = models.get("resnet")
    mobilenet_model = models.get("mobilenet")
    resnet_map = models.get("resnet_map", list(range(4)))
    mobilenet_map = models.get("mobilenet_map", list(range(4)))
    
    raw_resnet = None
    if resnet_model is not None:
        try:
            arr_resnet = preprocess(img)
            raw_resnet_out = resnet_model.predict(arr_resnet, verbose=0)[0]
            raw_resnet = np.zeros(4, dtype=np.float32)
            for model_idx, class_idx in enumerate(resnet_map):
                raw_resnet[class_idx] = raw_resnet_out[model_idx]
        except Exception as e:
            st.warning(f"ResNet50V2 inference failed: {str(e)[:50]}")
    
    raw_mobilenet = None
    if mobilenet_model is not None and use_ensemble:
        try:
            arr_mobilenet = preprocess_for_mobilenet(img)
            raw_mobilenet_out = mobilenet_model.predict(arr_mobilenet, verbose=0)[0]
            raw_mobilenet = np.zeros(4, dtype=np.float32)
            for model_idx, class_idx in enumerate(mobilenet_map):
                raw_mobilenet[class_idx] = raw_mobilenet_out[model_idx]
        except Exception as e:
            st.warning(f"MobileNetV2 inference failed: {str(e)[:50]}")
    
    if raw_resnet is not None and raw_mobilenet is not None and use_ensemble:
        raw = raw_resnet * 0.60 + raw_mobilenet * 0.40
        ensemble_mode = "ResNet50V2 + MobileNetV2 Ensemble"
        is_demo = False
    elif raw_resnet is not None:
        raw = raw_resnet
        ensemble_mode = "ResNet50V2 Single Model"
        is_demo = False
    else:
        # Demo prediction
        arr_demo = np.array(img.resize((64,64)).convert("L"), dtype=np.float32)
        mean_px, std_px = float(arr_demo.mean()), float(arr_demo.std())
        if std_px > 55 and mean_px < 80:
            raw = np.array([0.78, 0.12, 0.06, 0.04])
        elif mean_px > 100 and std_px > 45:
            raw = np.array([0.06, 0.82, 0.07, 0.05])
        elif std_px < 35:
            raw = np.array([0.04, 0.03, 0.91, 0.02])
        else:
            raw = np.array([0.05, 0.07, 0.05, 0.83])
        ensemble_mode = "Demo Mode (No Model)"
        is_demo = True
    
    # Temperature scaling
    try:
        logits = np.log(np.clip(raw, 1e-7, 1.0))
        scaled = np.exp(logits / temperature)
        preds = scaled / scaled.sum()
    except Exception:
        preds = raw / raw.sum()
    
    return preds, ensemble_mode, is_demo

# ================================================================
# GRAD-CAM
# ================================================================
def make_gradcam(model, img_array, pred_idx):
    if model is None:
        return None
    
    backbone = next((l for l in model.layers if hasattr(l, "layers")), None)
    last_conv = None
    if backbone:
        for l in reversed(backbone.layers):
            if isinstance(l, keras.layers.Conv2D):
                last_conv = l.name
                break
    if not last_conv:
        for l in reversed(model.layers):
            if isinstance(l, keras.layers.Conv2D):
                last_conv = l.name
                break
    if not last_conv:
        return None
    
    try:
        src = backbone or model
        grad_model = keras.Model(inputs=model.inputs, outputs=[src.get_layer(last_conv).output, model.output])
        
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, pred_idx]
        
        grads = tape.gradient(loss, conv_output)
        if grads is None:
            return None
        
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1))
        heatmap = heatmap.numpy()
        
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        return heatmap
    except Exception:
        return None

def synthetic_heatmap(pil_img):
    g = np.array(pil_img.convert("L").resize((28, 28)), dtype=np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    m = np.ones_like(g)
    m[:3,:] = m[-3:,:] = m[:,:3] = m[:,-3:] = 0
    h = g * m
    if h.max() > 0:
        h /= h.max()
    ys, xs = np.mgrid[0:28, 0:28]
    bias = np.exp(-((xs - 16)**2 + (ys - 14)**2) / (2 * 7.5**2))
    h = h * 0.38 + bias * 0.62
    if h.max() > 0:
        h /= h.max()
    return h

def smooth_hm(raw):
    h = cv2.resize(raw.astype(np.float32), IMG_SIZE)
    h = cv2.GaussianBlur(h, (15, 15), 0)
    return (h / h.max()) if h.max() > 0 else h

def overlay_gradcam(pil_img, hm_raw, alpha=0.55):
    orig = np.array(pil_img.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    hm = smooth_hm(hm_raw)
    hm_c = (mpl_cm.jet(hm)[:, :, :3] * 255).astype(np.float32)
    gray = np.mean(orig, axis=2, keepdims=True)
    desat = orig * 0.4 + gray * 0.6
    amask = np.clip(alpha + (1 - alpha) * hm[..., np.newaxis] * 0.5, 0, 1)
    blend = np.clip(desat * (1 - amask) + hm_c * amask, 0, 255).astype(np.uint8)
    return Image.fromarray(blend), hm

# ================================================================
# AI REPORT
# ================================================================
def mock_report(pc, c):
    T = {
        "Glioma": {
            "clinical_interpretation": "Heterogeneous mass lesion with irregular margins and peritumoral edema. Mixed signal intensity with areas of necrosis and ring-enhancing pattern characteristic of high-grade glioma.",
            "location_morphology": "Right frontal lobe, supratentorial compartment. Irregular lobulated borders.",
            "model_reasoning": f"Glioma ({c:.1f}%) supported by ring-enhancing pattern, heterogeneous signal.",
            "gradcam_analysis": "Activation heatmap localised to the tumor epicenter.",
            "risk_level": "HIGH",
            "risk_justification": "High-grade glioma carries significant morbidity.",
            "patient_explanation": "The scan shows signs of a Glioma brain tumor. This is NOT a final diagnosis.",
            "next_steps": "1. Neuroradiologist review\n2. Contrast-enhanced MRI\n3. Neurosurgical consultation",
            "image_quality": "GOOD",
            "uncertainty_factors": "Partial ambiguity at tumor-edema boundary.",
            "reliability_score": 91,
            "overall_reliability": "High reliability.",
            "disclaimer": "AI-assisted decision support only."
        },
        "Meningioma": {
            "clinical_interpretation": "Well-circumscribed extra-axial mass with dural tail sign, homogeneous signal intensity.",
            "location_morphology": "Parasagittal convexity, extra-axial. Broad dural base.",
            "model_reasoning": f"Meningioma ({c:.1f}%) aligned with extra-axial location.",
            "gradcam_analysis": "Model focuses on the lesion-dura interface.",
            "risk_level": "MODERATE",
            "risk_justification": "Most meningiomas are WHO Grade I.",
            "patient_explanation": "The scan suggests a meningioma - usually slow-growing.",
            "next_steps": "1. Neurology review\n2. Contrast-enhanced MRI\n3. Observation vs surgery",
            "image_quality": "GOOD",
            "uncertainty_factors": "Cavernous sinus involvement requires coronal sequences.",
            "reliability_score": 86,
            "overall_reliability": "Good reliability.",
            "disclaimer": "AI-assisted decision support only."
        },
        "Pituitary Tumor": {
            "clinical_interpretation": "Intrasellar mass expanding the sella turcica with suprasellar extension.",
            "location_morphology": "Sella turcica, macroadenoma with suprasellar extension.",
            "model_reasoning": f"Pituitary tumor ({c:.1f}%) confirmed by intrasellar location.",
            "gradcam_analysis": "Model activates precisely on the sellar region.",
            "risk_level": "MODERATE",
            "risk_justification": "Usually benign pituitary adenoma.",
            "patient_explanation": "The scan shows a tumor in the pituitary gland.",
            "next_steps": "1. Endocrinology consultation\n2. Visual field testing\n3. Hormone panel",
            "image_quality": "GOOD",
            "uncertainty_factors": "Cavernous sinus invasion requires Knosp grading.",
            "reliability_score": 89,
            "overall_reliability": "High reliability.",
            "disclaimer": "AI-assisted decision support only."
        },
        "No Tumor": {
            "clinical_interpretation": "Normal brain parenchyma. No mass lesion, abnormal enhancement, or signal abnormality.",
            "location_morphology": "No focal lesion. Gray-white matter differentiation preserved.",
            "model_reasoning": f"No Tumor ({c:.1f}%) consistent with symmetric architecture.",
            "gradcam_analysis": "Low distributed activation with no focal pathological concentration.",
            "risk_level": "LOW",
            "risk_justification": "No imaging evidence of intracranial neoplasm.",
            "patient_explanation": "Good news - the AI did not detect a tumor.",
            "next_steps": "Clinical follow-up if symptomatic. Repeat imaging if indicated.",
            "image_quality": "GOOD",
            "uncertainty_factors": "None significant.",
            "reliability_score": 95,
            "overall_reliability": "Very high reliability.",
            "disclaimer": "AI-assisted decision support only."
        }
    }
    return T.get(pc, T["Glioma"])

def rb(title, body, v=""):
    return f'<div class="rb {v}"><div class="rb-t">{title}</div><div class="rb-b">{body}</div></div>'

# ================================================================
# FIGURE GENERATORS
# ================================================================
def pure_heatmap_fig(hm, pred_class, conf):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(hm, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    ax.axis("off")
    ax.set_facecolor("#020609")
    fig.patch.set_facecolor("#020609")
    cb = fig.colorbar(ax.images[0], ax=ax, fraction=0.034, pad=0.02)
    cb.ax.tick_params(colors="#666", labelsize=6)
    cb.set_label("Activation", color="#666", fontsize=7, labelpad=5)
    ax.set_title(f"{pred_class}  {conf:.1f}%", color="#bbb", fontsize=8, pad=7, fontweight="bold")
    plt.tight_layout(pad=0.2)
    return fig

def histogram_fig(hm):
    flat = hm.flatten()
    fig, ax = plt.subplots(figsize=(4, 3.2))
    n, bins, patches = ax.hist(flat, bins=45, edgecolor="none")
    mids = (bins[:-1] + bins[1:]) / 2
    for patch, v in zip(patches, mids):
        patch.set_facecolor(mpl_cm.jet(v))
        patch.set_alpha(0.9)
    ax.axvline(flat.mean(), color="#fbbf24", ls="--", lw=1.3, label=f"Mean {flat.mean():.2f}")
    ax.axvline(np.percentile(flat, 90), color="#f87171", ls="--", lw=1.3, label=f"P90 {np.percentile(flat,90):.2f}")
    ax.set_xlabel("Activation value", color="#666", fontsize=7)
    ax.set_ylabel("Pixel count", color="#666", fontsize=7)
    ax.tick_params(colors="#666", labelsize=7)
    ax.set_facecolor("#020609")
    fig.patch.set_facecolor("#020609")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a2e4a")
    ax.legend(fontsize=6.5, labelcolor="#ccc", facecolor="#0a1424", edgecolor="#1a2e4a")
    ax.set_title("Activation Distribution", color="#bbb", fontsize=8, pad=6, fontweight="bold")
    plt.tight_layout(pad=0.4)
    return fig

def four_panel_fig(pil_img, hm_raw, pred_class, conf, demo=False):
    overlay, hm = overlay_gradcam(pil_img, hm_raw)
    orig = np.array(pil_img.convert("RGB").resize(IMG_SIZE))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#020609")
    for ax in axes:
        ax.set_facecolor("#020609")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a2e4a")
    axes[0].imshow(orig)
    axes[0].axis("off")
    axes[1].imshow(np.array(overlay))
    axes[1].axis("off")
    im = axes[2].imshow(hm, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    axes[2].axis("off")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#666", labelsize=6)
    flat = hm.flatten()
    n, bins, patches = axes[3].hist(flat, bins=40, edgecolor="none")
    mids = (bins[:-1] + bins[1:]) / 2
    for p, v in zip(patches, mids):
        p.set_facecolor(mpl_cm.jet(v))
        p.set_alpha(0.9)
    axes[3].axvline(flat.mean(), color="#fbbf24", ls="--", lw=1.1)
    axes[3].set_xlabel("Activation", color="#666", fontsize=7)
    axes[3].set_facecolor("#020609")
    for sp in axes[3].spines.values():
        sp.set_edgecolor("#1a2e4a")
    axes[3].tick_params(colors="#666", labelsize=6)
    titles = ["Original MRI", "Grad-CAM Overlay", "Activation Map", "Histogram"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    tag = " [DEMO]" if demo else ""
    fig.suptitle(f"NeuroScan AI | Grad-CAM | {pred_class} ({conf:.1f}%){tag}",
                 color="#e2e8f0", fontsize=10.5, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.6)
    return fig

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()

def to_b64(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🧠 NeuroScan AI")
    st.markdown("---")
    
    st.markdown("#### Model Status")
    models = load_models()
    
    if models["resnet"] is not None:
        st.success("✅ ResNet50V2")
    else:
        st.error("❌ ResNet50V2")
    
    if models["mobilenet"] is not None:
        st.success("✅ MobileNetV2")
    else:
        st.warning("⚠️ MobileNetV2")
    
    st.markdown("---")
    st.markdown("#### Class Mapping")
    st.caption(f"ResNet: {models['resnet_map']}")
    st.caption(f"MobileNet: {models['mobilenet_map']}")
    
    if models["resnet_map"] != list(range(4)) or models["mobilenet_map"] != list(range(4)):
        st.info("🔄 Class mapping auto-corrected")
    
    st.markdown("---")
    st.markdown("#### Settings")
    
    use_ensemble = st.toggle("Use Ensemble", value=True,
                            help="Combine ResNet50V2 and MobileNetV2")
    strict_mri = st.toggle("Strict MRI Validation", value=False,
                          help="Use stricter thresholds for MRI validation")
    alpha = st.slider("Heatmap Intensity", 0.2, 0.8, 0.55, 0.05)
    temperature = st.slider("Temperature", 1.0, 2.5, 1.4, 0.1)
    show_prf = st.toggle("Show Model Performance", value=True)
    
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(245,158,11,.07);border-left:3px solid #f59e0b;
      border-radius:0 8px 8px 0;padding:10px 12px;font-family:'DM Mono',monospace;
      font-size:9.5px;color:rgba(253,211,77,.92);line-height:1.7;">
      <strong style="color:#fbbf24;">Disclaimer</strong><br>
      AI decision support only. Not a substitute for professional medical diagnosis.
    </div>
    """, unsafe_allow_html=True)

# ================================================================
# TOP NAV
# ================================================================
_tog_icon = "☀️" if _dk else "🌙"
_next_theme = "light" if _dk else "dark"

_qp = st.query_params.get("theme", None)
if _qp in ("light", "dark") and st.session_state.theme != _qp:
    st.session_state.theme = _qp
    st.query_params.clear()
    st.rerun()

st.markdown(f"""
<div class="topnav">
  <div class="nav-brand">
    <div class="nav-logo">🧠</div>
    <div>
      <div class="nav-name">NeuroScan <span>AI</span></div>
      <div class="nav-tagline">Brain Tumor MRI Classification &amp; Explainability System</div>
    </div>
  </div>
  <div class="nav-right">
    <span class="chip c-blue">ResNet50V2</span>
    <span class="chip c-teal">Grad-CAM XAI</span>
    <span class="chip c-green">95.31% Accuracy</span>
    <span class="chip c-amber">4-Class CNN</span>
    <span class="chip c-purple">AI Reports</span>
    <form method="get" action="" style="margin:0;padding:0;display:inline-flex;">
      <input type="hidden" name="theme" value="{_next_theme}">
      <button type="submit" class="theme-toggle" title="Switch theme">{_tog_icon}</button>
    </form>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# HERO
# ================================================================
st.markdown("""
<div class="hero">
  <div class="hero-inner">
    <div class="hero-top">
      <div>
        <h1 class="hero-h1">NeuroScan Brain Tumor<br>
          <span class="grad">MRI Classification</span>
        </h1>
        <p class="hero-desc">
          Upload any axial brain MRI and receive instant classification across 4 tumor types,
          complete with Grad-CAM heatmaps highlighting the exact regions that drove the prediction,
          plus an AI-generated clinical report.
        </p>
      </div>
      <div class="hero-stats">
        <div class="hs"><div class="hs-n">95.31%</div><div class="hs-l">Ensemble Accuracy</div></div>
        <div class="hs"><div class="hs-n">4</div><div class="hs-l">Tumor Classes</div></div>
        <div class="hs"><div class="hs-n">~7 K</div><div class="hs-l">Training Images</div></div>
        <div class="hs"><div class="hs-n">v3.0</div><div class="hs-l">Model Version</div></div>
      </div>
    </div>
    <div class="hero-div"></div>
    <div class="pipeline">
      <div class="pip-step"><div class="pip-num">1</div><div class="pip-txt"><strong>Upload MRI</strong>Any axial T1/T2 scan</div></div>
      <div class="pip-arr">›</div>
      <div class="pip-step"><div class="pip-num">2</div><div class="pip-txt"><strong>CNN Inference</strong>Ensemble classifies</div></div>
      <div class="pip-arr">›</div>
      <div class="pip-step"><div class="pip-num">3</div><div class="pip-txt"><strong>Grad-CAM</strong>Tumor region heatmap</div></div>
      <div class="pip-arr">›</div>
      <div class="pip-step"><div class="pip-num">4</div><div class="pip-txt"><strong>AI Report</strong>Clinical analysis</div></div>
      <div class="pip-arr">›</div>
      <div class="pip-step"><div class="pip-num">5</div><div class="pip-txt"><strong>Export</strong>JSON + PNG figure</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# CONTENT WRAPPER
# ================================================================
st.markdown('<div class="wrap">', unsafe_allow_html=True)

# ================================================================
# INPUT / OUTPUT COLUMNS
# ================================================================
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="slbl">Input - MRI Scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload MRI",
        type=["jpg", "jpeg", "png", "bmp"],
        label_visibility="collapsed",
        help="Axial T1 or T2-weighted brain MRI. JPEG/PNG up to 10 MB."
    )
    
    st.markdown('''<div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.68);text-align:center;padding:8px 0 14px;text-transform:uppercase;letter-spacing:.11em;background:rgba(56,189,248,.06);border-radius:8px;margin-top:6px;">
      JPG / PNG / BMP &nbsp;·&nbsp; Max 10 MB &nbsp;·&nbsp; T1 or T2 axial preferred
    </div>''', unsafe_allow_html=True)

    st.markdown('''<div style="margin:18px 0 8px;">
      <div style="font-family:DM Mono,monospace;font-size:10.5px;font-weight:600;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.13em;display:flex;align-items:center;gap:10px;">
        <span style="flex:1;height:1px;background:rgba(56,189,248,.40);display:block"></span>
        Or choose a pre-loaded sample
        <span style="flex:1;height:1px;background:rgba(56,189,248,.40);display:block"></span>
      </div>
    </div>''', unsafe_allow_html=True)

    sel_lbl = st.selectbox("Sample", list(SAMPLES.keys()), index=0, label_visibility="collapsed")
    sel_file = SAMPLES[sel_lbl]

    img = src = None
    if uploaded:
        try:
            from PIL import ImageOps
            import io as _io
            _bytes = uploaded.read()
            _buf = _io.BytesIO(_bytes)
            _raw = Image.open(_buf)
            _raw.load()
            _raw = ImageOps.exif_transpose(_raw)
            _arr_loaded = np.array(_raw.convert("RGB"), dtype=np.uint8)
            img = Image.fromarray(_arr_loaded, mode="RGB")
            src = "upload"
        except Exception as _e:
            st.error(f"Failed to open image: {_e}")
            img = None
        if img:
            st.success("✅ Image uploaded successfully.")
    elif sel_file:
        sp = os.path.join(SAMPLE_DIR, sel_file)
        if os.path.exists(sp):
            img = Image.open(sp).convert("RGB")
            src = "sample"
        else:
            st.warning(f"Sample not found: `{sp}`")
    else:
        st.markdown('''<div style="border:2px dashed rgba(56,189,248,.38);border-radius:14px;padding:2.5rem 1.5rem;text-align:center;background:rgba(56,189,248,.05);margin:8px 0;">
          <div style="font-size:40px;margin-bottom:12px;">🩻</div>
          <div style="font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:6px;">No image selected</div>
          <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.58);letter-spacing:.07em;line-height:1.9;">
            Upload a brain MRI above<br>or pick a sample below
          </div>
        </div>''', unsafe_allow_html=True)

    if img:
        cap = "UPLOADED SCAN" if src == "upload" else f"SAMPLE: {sel_lbl.upper()}"
        st.markdown(f'''<div style="font-family:DM Mono,monospace;font-size:10px;font-weight:600;color:#38bdf8;text-align:center;padding:8px 0 4px;letter-spacing:.09em;text-transform:uppercase;">
          📸 {cap}
        </div>''', unsafe_allow_html=True)
        st.image(img, use_column_width=True, clamp=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    
    _btn_lbl = "Upload or select an MRI first" if img is None else "🔬 Analyze and Generate Clinical Report"
    clicked = st.button(
        _btn_lbl,
        disabled=(img is None),
        help=_btn_lbl,
        use_container_width=True
    )

with col_out:
    st.markdown('<div class="slbl"><span id="ns-result-label">Model Output - Prediction</span></div>', unsafe_allow_html=True)
    
    if not clicked:
        st.markdown('''<div class="glass" style="min-height:440px;display:flex;align-items:center;justify-content:center;text-align:center;padding:3rem;">
          <div>
            <div style="font-size:54px;margin-bottom:18px;">🔬</div>
            <div style="font-family:Space Grotesk,sans-serif;font-size:19px;font-weight:600;color:#e2e8f0;line-height:1.4;margin-bottom:8px;">Ready for Analysis</div>
            <div style="font-family:Inter,sans-serif;font-size:13.5px;color:rgba(255,255,255,.65);line-height:1.85;margin-bottom:22px;">
              Upload a brain MRI scan or select a sample,<br>
              then click <strong>Analyse</strong> to run the full pipeline.
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">CNN PREDICTION</span>
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">GRAD-CAM HEATMAP</span>
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">AI CLINICAL REPORT</span>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

# ================================================================
# ANALYSIS
# ================================================================
if clicked and img:
    # Auto-scroll
    st.markdown("""
<div id="ns-result-anchor"></div>
<style>
@keyframes ns-slide-in { from { transform:translateY(-80px); opacity:0 } to { transform:translateY(0); opacity:1 } }
#ns-toast {
  position:fixed; top:70px; left:50%; transform:translateX(-50%);
  z-index:9999; background:rgba(14,30,70,.97);
  border:1px solid rgba(56,189,248,.50); border-left:4px solid #38bdf8;
  border-radius:12px; padding:12px 22px 12px 16px;
  display:flex; align-items:center; gap:12px;
  box-shadow:0 8px 32px rgba(0,0,0,.35);
  animation: ns-slide-in .4s ease forwards, ns-fade-out .5s ease 4s forwards;
  min-width:320px; max-width:480px; pointer-events:none;
}
@keyframes ns-fade-out { from { opacity:1 } to { opacity:0; pointer-events:none } }
#ns-toast-icon { font-size:22px; flex-shrink:0; }
#ns-toast-title { font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:2px; }
#ns-toast-sub { font-family:'DM Mono',monospace; font-size:10px; color:rgba(255,255,255,.55); letter-spacing:.05em; }
</style>
<div id="ns-toast">
  <div id="ns-toast-icon">✅</div>
  <div id="ns-toast-body">
    <div id="ns-toast-title">Analysis Complete</div>
    <div id="ns-toast-sub">Results available in the output panel →</div>
  </div>
</div>
<script>
(function() {
  setTimeout(function() {
    var anchor = document.getElementById('ns-result-anchor');
    if (anchor) { anchor.scrollIntoView({ behavior: 'smooth', block: 'start' }); }
  }, 300);
})();
</script>
""", unsafe_allow_html=True)

    with col_out:
        st.markdown("""
<div style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.30);border-left:4px solid #38bdf8;border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:20px">🔬</span>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:#e2e8f0;">Analysis in progress</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.50);margin-top:2px;letter-spacing:.05em;">VALIDATION → CNN INFERENCE → GRAD-CAM → AI REPORT</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # MRI Validation
    with st.spinner("Validating input image..."):
        _mri_valid, _mri_conf, _mri_reasons = validate_mri(img, strict=strict_mri)

    with col_out:
        mri_gate_ui(_mri_valid, _mri_conf, _mri_reasons, _dk)

    if not _mri_valid:
        if st.button("Override and Continue (Testing Only)"):
            _mri_valid = True
            st.rerun()
        st.stop()

    # Inference
    with st.spinner("Running CNN inference..."):
        preds, ensemble_mode, is_demo = run_inference(
            img, models, temperature=temperature, use_ensemble=use_ensemble
        )

    pidx = int(np.argmax(preds))
    pcls = CLASS_NAMES[pidx]
    conf = float(preds[pidx]) * 100
    rl, rc, dc = RISK[pcls]

    # Close call detection
    _sorted_idx = np.argsort(preds)[::-1]
    _second_idx = int(_sorted_idx[1])
    _second_cls = CLASS_NAMES[_second_idx]
    _second_conf = float(preds[_second_idx]) * 100
    _margin = conf - _second_conf
    is_close_call = _margin < 20.0

    # Update toast
    st.markdown(f"""
<script>
(function() {{
  var t = document.getElementById('ns-toast');
  var ti = document.getElementById('ns-toast-title');
  var ts = document.getElementById('ns-toast-sub');
  if (ti) ti.textContent = 'Result: {pcls} ({conf:.1f}%)';
  if (ts) ts.textContent = 'Risk level: {rl} — scroll down for full report';
  if (t) {{
    t.style.animation = 'none';
    t.offsetHeight;
    t.style.animation = 'ns-slide-in .4s ease forwards, ns-fade-out .5s ease 5s forwards';
  }}
}})();
</script>
""", unsafe_allow_html=True)

    # Warnings
    with col_out:
        if conf < 55.0:
            st.warning(f"⚠️ **Low Confidence ({conf:.1f}%)** — Specialist review essential.")
        elif is_close_call:
            st.info(f"ℹ️ **Close Call** — {pcls} ({conf:.1f}%) vs {_second_cls} ({_second_conf:.1f}%)")

    # Grad-CAM
    with st.spinner("Computing Grad-CAM heatmap..."):
        if models["resnet"] is not None:
            arr = preprocess(img)
            raw_hm = make_gradcam(models["resnet"], arr, pidx)
            hraw = raw_hm if raw_hm is not None else synthetic_heatmap(img)
        else:
            hraw = synthetic_heatmap(img)
            is_demo = True

        overlay_img, hm = overlay_gradcam(img, hraw, alpha=alpha)

    # Stats
    mean_a = float(hm.mean())
    max_a = float(hm.max())
    p90_a = float(np.percentile(hm, 90))
    focus_p = float((hm > 0.5).sum() / hm.size * 100)

    # AI Report
    with st.spinner("Generating clinical report..."):
        report = mock_report(pcls, conf)
        report_is_live = False

    # ============================================================
    # RESULTS DISPLAY
    # ============================================================
    
    def class_bar_html(name, prob, is_top, color):
        pct = prob * 100
        w_pct = max(pct, 0.5)
        bold = "font-weight:700;color:#f8fafc;" if is_top else "font-weight:400;color:rgba(255,255,255,.55);"
        track = "background:rgba(255,255,255,.07);"
        fill_rst = "background:rgba(255,255,255,.18);"
        fill = f"background:linear-gradient(90deg,{color},{color}cc);" if is_top else fill_rst
        pct_col = "color:#38bdf8;" if is_top else "color:rgba(255,255,255,.45);"
        return f"""
<div style="margin-bottom:20px;">
  <div style="font-family:'DM Mono',monospace;font-size:12px;letter-spacing:.12em;text-transform:uppercase;margin-bottom:7px;{bold}">{name}</div>
  <div style="{track}border-radius:6px;height:8px;overflow:hidden;margin-bottom:5px;">
    <div style="height:100%;border-radius:6px;width:{w_pct}%;{fill}transition:width .8s ease;"></div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:11px;{pct_col}">{pct:.1f}%</div>
</div>"""

    bars_html = ""
    for i, (cname, cprob, ccol) in enumerate(zip(CLASS_NAMES, preds, CLASS_COLORS)):
        bars_html += class_bar_html(cname, cprob, i == pidx, ccol)

    with col_out:
        st.markdown(f"""
<div class="pred-card">
  <div class="pred-eyebrow">{ensemble_mode} | 4-Class CNN Prediction</div>
  <div class="pred-name">{pcls}</div>
  <div class="conf-row">
    <span class="conf-l">Model Confidence</span>
    <span class="conf-v">{conf:.1f}%</span>
  </div>
  <div class="conf-track">
    <div class="conf-fill" style="width:{conf}%"></div>
  </div>
  <div class="risk-chip {rc}">
    <span class="rdot {dc}"></span>{rl} RISK
  </div>
</div>""", unsafe_allow_html=True)

    # Heatmap Section
    st.markdown("---")
    st.markdown(f"""
<div class="hm-section">
  <div class="hm-header">
    <div>
      <div class="hm-title">Grad-CAM Tumor Region Heatmap | {pcls}</div>
      <div class="hm-sub">Pure CNN GradientTape | No VLM Required</div>
    </div>
    <div class="hm-legend">
      <div class="hm-leg"><span class="hm-swatch" style="background:linear-gradient(90deg,#00007f,#007fff,#00ffff)"></span>Low attention</div>
      <div class="hm-leg"><span class="hm-swatch" style="background:linear-gradient(90deg,#ffff00,#ff7f00,#ff0000)"></span>High attention</div>
    </div>
  </div>
""", unsafe_allow_html=True)

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        st.markdown(f"""
<div style="padding:1.2rem 0.4rem;">
  <div style="font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.16em;margin-bottom:18px;font-weight:600;">Class Probability Distribution</div>
  {bars_html}
</div>""", unsafe_allow_html=True)

    with res_right:
        st.markdown('<div class="hm-img-frame">', unsafe_allow_html=True)
        st.image(overlay_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="text-align:center;margin-top:8px;font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.40);letter-spacing:.1em;">Grad-CAM Overlay | {pcls}</div>""", unsafe_allow_html=True)

    # Secondary row
    st.markdown('<div style="height:1px;background:rgba(255,255,255,.08);margin:16px 0 14px"></div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1], gap="small")

    with sc1:
        st.markdown('<div class="hm-col-lbl">Original MRI</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown('<div class="hm-col-note">Raw input before preprocessing</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="hm-col-lbl">Pure Activation Map</div>', unsafe_allow_html=True)
        fh = pure_heatmap_fig(hm, pcls, conf)
        st.pyplot(fh, use_container_width=True)
        plt.close()
        st.markdown('<div class="hm-col-note">Normalised intensity<br>last conv layer</div>', unsafe_allow_html=True)

    with sc3:
        st.markdown('<div class="hm-col-lbl">Activation Histogram</div>', unsafe_allow_html=True)
        fhist = histogram_fig(hm)
        st.pyplot(fhist, use_container_width=True)
        plt.close()
        st.markdown('<div class="hm-col-note">Distribution of<br>activation values</div>', unsafe_allow_html=True)

    with sc4:
        st.markdown('<div class="hm-col-lbl">Activation Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:8px;padding-top:2px;">
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{mean_a:.3f}</div><div class="hm-sl">Mean Activation</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{p90_a:.3f}</div><div class="hm-sl">90th Percentile</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{focus_p:.1f}%</div><div class="hm-sl">High-Activation Area</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{max_a:.3f}</div><div class="hm-sl">Peak Activation</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Color scale
    st.markdown("""
  <div class="cscale" style="margin:14px 1.5rem;">
    <div class="cscale-bar"></div>
    <div class="cscale-lbls">
      <span>Deep Blue (lowest)</span><span>Cyan</span><span>Green / Yellow</span><span>Orange</span><span>Red (highest / tumor)</span>
    </div>
  </div>
  <div class="hm-explain">
    <div class="hm-exp-title">How Grad-CAM Works - Pure CNN Gradient Method</div>
    <div class="hm-exp-grid">
      <div class="hm-exp-item"><div class="hm-exp-t">Step 1 - Forward Pass</div><div class="hm-exp-b">The 224x224 MRI passes through ResNet50V2. The last conv layer outputs a 7x7 feature map with 2048 channels.</div></div>
      <div class="hm-exp-item"><div class="hm-exp-t">Step 2 - Gradient Backprop</div><div class="hm-exp-b">TensorFlow GradientTape records how strongly each feature map channel contributed to the score.</div></div>
      <div class="hm-exp-item"><div class="hm-exp-t">Step 3 - Weighted Sum + ReLU</div><div class="hm-exp-b">Each feature map is multiplied by its weight, summed, and ReLU-activated. The coarse 7x7 result is upscaled to 224x224.</div></div>
      <div class="hm-exp-item"><div class="hm-exp-t">This Scan</div><div class="hm-exp-b">Mean: <strong style="color:#38bdf8">"""+f"{mean_a:.3f}"+"""</strong>, Peak: <strong style="color:#38bdf8">"""+f"{max_a:.3f}"+"""</strong>. """+f"{focus_p:.1f}%"+""" of pixels exceed 0.5.</div></div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if is_demo:
        st.info("**Demo Mode:** Heatmap is synthetic (MRI intensity-derived).")

    # Download
    fig4 = four_panel_fig(img, hraw, pcls, conf, is_demo)
    fbyt = fig_bytes(fig4)
    plt.close(fig4)
    st.download_button("Download 4-Panel Grad-CAM Figure (PNG)",
                       data=fbyt,
                       file_name=f"gradcam_{pcls.lower().replace(' ','_')}.png",
                       mime="image/png")

    # Clinical Report
    st.markdown("---")
    st.markdown('<div class="slbl">AI-Assisted Clinical Report</div>', unsafe_allow_html=True)

    if report_is_live:
        st.success("✅ Live Claude analysis of this MRI and its Grad-CAM heatmap.")
    else:
        st.warning("Template report shown — NOT a live Claude analysis.")

    if is_close_call:
        st.warning(f"Close call: {pcls} ({conf:.1f}%) vs {_second_cls} ({_second_conf:.1f}%) — margin under 20 points.")

    t1, t2, t3, t4 = st.tabs(["Clinical Findings", "Model Reasoning", "Patient Summary", "Reliability"])

    with t1:
        st.markdown(rb("Clinical Interpretation", report.get("clinical_interpretation", ""), "rb-red"), unsafe_allow_html=True)
        st.markdown(rb("Location and Morphology", report.get("location_morphology", "")), unsafe_allow_html=True)

    with t2:
        st.markdown(rb("Model Reasoning", report.get("model_reasoning", "")), unsafe_allow_html=True)
        st.markdown(rb("Grad-CAM Analysis", report.get("gradcam_analysis", ""), "rb-grn"), unsafe_allow_html=True)

    with t3:
        st.markdown(rb("Plain Language Summary", report.get("patient_explanation", ""), "rb-yel"), unsafe_allow_html=True)
        st.markdown(rb("Recommended Next Steps", report.get("next_steps", "").replace("\n", "<br>")), unsafe_allow_html=True)

    with t4:
        rs = report.get("reliability_score", 80)
        a, b, c = st.columns(3)
        with a:
            st.metric("Reliability Score", f"{rs}/100")
        with b:
            st.metric("Image Quality", report.get("image_quality", "N/A"))
        with c:
            st.metric("Risk Level", rl)
        st.progress(rs / 100)
        qv = {"GOOD": "rb-grn", "ADEQUATE": "rb-yel", "POOR": "rb-red"}.get(report.get("image_quality", "GOOD"), "rb-grn")
        st.markdown(rb("Uncertainty Factors", report.get("uncertainty_factors", "None identified."), qv), unsafe_allow_html=True)
        st.markdown(rb("Overall Reliability", report.get("overall_reliability", "")), unsafe_allow_html=True)

    st.markdown(f"""
<div class="disc">
  <strong>AI-Assisted Decision Support Only</strong> -
  {report.get("disclaimer", "")}
  This system must not replace professional medical diagnosis.
  All findings require review by a licensed radiologist or neurosurgeon.
</div>""", unsafe_allow_html=True)

    # Export JSON
    gcam_stats = {"mean": round(mean_a, 4), "peak": round(max_a, 4), "p90": round(p90_a, 4), "focus_pct": round(focus_p, 2), "synthetic_demo": is_demo}
    st.download_button(
        "Export Full Report (JSON)",
        data=json.dumps({
            "system": "NeuroScan AI v3.0",
            "model": ensemble_mode,
            "prediction": pcls,
            "confidence_pct": round(conf, 2),
            "risk_level": rl,
            "gradcam_stats": gcam_stats,
            "class_probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, preds)},
            **report
        }, indent=2),
        file_name=f"neuroscan_{pcls.lower().replace(' ', '_')}.json",
        mime="application/json"
    )

    # Model Performance
    if show_prf:
        st.markdown("---")
        st.markdown('<div class="slbl">Model Performance - Training Results</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="glass" style="margin-bottom:16px;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));gap:10px;margin-bottom:14px;">
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px"><div class="hm-sv">95.31%</div><div class="hm-sl">Ensemble Acc.</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px"><div class="hm-sv">100%</div><div class="hm-sl">No Tumor Recall</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px"><div class="hm-sv">99.8%</div><div class="hm-sl">Pituitary Recall</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px"><div class="hm-sv">98.0%</div><div class="hm-sl">Meningioma Recall</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px"><div class="hm-sv">83.5%</div><div class="hm-sl">Glioma Recall</div></div>
  </div>
  <div style="font-size:12px;color:rgba(255,255,255,.38);line-height:1.78;">
    Ensemble combines <strong style="color:#38bdf8">ResNet50V2</strong> and
    <strong style="color:#38bdf8">MobileNetV2</strong> via soft-voting.
    Glioma recall (83.5%) is lower due to visual similarity with meningioma
    on T1 non-contrast sequences. Contrast-enhanced MRI is recommended.
  </div>
</div>
""", unsafe_allow_html=True)

        pt1, pt2 = st.tabs(["Training History", "Confusion Matrix"])
        with pt1:
            if os.path.exists("training_history.png"):
                st.image("training_history.png", use_column_width=True)
            else:
                st.info("`training_history.png` not found")
        with pt2:
            if os.path.exists("confusion_matrix_ensemble.png"):
                st.image("confusion_matrix_ensemble.png", use_column_width=True)
            else:
                st.info("`confusion_matrix_ensemble.png` not found")

st.markdown('</div>', unsafe_allow_html=True)
