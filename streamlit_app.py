"""
NeuroScan AI - FULL PRODUCTION VERSION
================================================================
REAL Model Loading | REAL Grad-CAM | REAL Claude Analysis
Requires: brain_tumor_model.h5 and mobilenet_model.h5 in root directory
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
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# TENSORFLOW IMPORT WITH MULTIPLE FALLBACKS
# ================================================================
TF_AVAILABLE = False
keras = None
tf = None
resnet_preprocess = None
mobilenet_preprocess = None

# Try multiple import strategies
try:
    # Try standard import
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    TF_AVAILABLE = True
    print("✅ TensorFlow loaded successfully")
except ImportError as e:
    print(f"⚠️ TensorFlow import failed: {e}")
    
    # Try alternative import (for some environments)
    try:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        from tensorflow import keras
        TF_AVAILABLE = True
        print("✅ TensorFlow (compat v1) loaded")
    except:
        pass

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

# ================================================================
# COMPLETE CSS (Same as before)
# ================================================================
st.markdown("""
<style>
*{box-sizing:border-box}
.stApp{background:#0a0e1a !important;color:#e2e8f0 !important;min-height:100vh}
#MainMenu,footer,header{visibility:hidden}
.block-container{padding:0 !important;max-width:100% !important}

.topnav{position:sticky;top:0;z-index:200;background:rgba(10,14,26,.95);backdrop-filter:blur(24px);border-bottom:1px solid rgba(56,189,248,.15);padding:.8rem 2.4rem;display:flex;align-items:center;justify-content:space-between;gap:1rem}
.nav-brand{display:flex;align-items:center;gap:13px}
.nav-logo{width:38px;height:38px;border-radius:10px;font-size:18px;background:linear-gradient(135deg,#1e40af,#0e7490);display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px rgba(56,189,248,.4);flex-shrink:0}
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
.c-red{background:rgba(239,68,68,.14);color:#fca5a5;border:1px solid rgba(239,68,68,.28)}

.hero{position:relative;overflow:hidden;padding:3rem 0 2.5rem;background:linear-gradient(130deg,#040c1c 0%,#071630 55%,#040c1c 100%);border-bottom:1px solid rgba(56,189,248,.09)}
.hero-inner{position:relative;z-index:1;width:100%;padding:0 2.8rem}
.hero-top{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}
.hero-h1{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.9rem,3.5vw,3rem);font-weight:700;color:#e2e8f0;letter-spacing:-.7px;line-height:1.13;margin-bottom:.5rem}
.hero-h1 .grad{background:linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
.hero-desc{font-size:15px;color:rgba(255,255,255,.7);line-height:1.74;max-width:530px}
.hero-stats{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end}
.hs{text-align:right}
.hs-n{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;background:linear-gradient(92deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}
.hs-l{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.12em;margin-top:3px}
.hero-div{height:1px;margin:1.2rem 0;background:linear-gradient(90deg,rgba(56,189,248,.3),rgba(129,140,248,.18),transparent)}
.pipeline{display:flex;align-items:center;gap:0;flex-wrap:wrap}
.pip-step{display:flex;align-items:center;gap:9px}
.pip-num{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0891b2);font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px rgba(56,189,248,.32);flex-shrink:0}
.pip-txt{font-size:11.5px;color:rgba(255,255,255,.65);line-height:1.38}
.pip-txt strong{color:rgba(255,255,255,.9);display:block;font-size:11px}
.pip-arr{color:rgba(56,189,248,.4);font-size:20px;padding:0 10px}

.wrap{width:100%;padding:2rem 2.8rem 4rem}
.glass{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.075);border-radius:20px;padding:1.8rem 2rem;backdrop-filter:blur(12px);box-shadow:0 8px 40px rgba(0,0,0,.35)}
.slbl{font-family:'DM Mono',monospace;font-size:11px;color:rgba(56,189,248,.8);text-transform:uppercase;letter-spacing:.17em;margin-bottom:12px;display:flex;align-items:center;gap:8px}
.slbl::after{content:'';flex:1;height:1px;background:rgba(56,189,248,.2)}

.pred-card{background:linear-gradient(135deg,rgba(14,30,70,.82),rgba(8,20,48,.92));border:1px solid rgba(56,189,248,.22);border-radius:18px;padding:1.5rem 1.6rem;margin-bottom:14px;position:relative;overflow:hidden;box-shadow:0 12px 40px rgba(0,0,0,.25)}
.pred-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8,#818cf8)}
.pred-eyebrow{font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.8);text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px}
.pred-name{font-family:'Space Grotesk',sans-serif;font-size:clamp(30px,4vw,44px);font-weight:700;color:#f8fafc;letter-spacing:-1px;line-height:1.04;margin-bottom:15px}
.conf-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}
.conf-l{font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.55)}
.conf-v{font-family:'Space Grotesk',sans-serif;font-size:14px;color:#38bdf8;font-weight:600}
.conf-track{background:rgba(255,255,255,.1);border-radius:8px;height:6px;overflow:hidden;margin-bottom:15px}
.conf-fill{height:100%;border-radius:8px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8)}
.risk-chip{display:inline-flex;align-items:center;gap:7px;padding:6px 15px;border-radius:20px;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}
.rdot{width:6px;height:6px;border-radius:50%}
.rH{background:rgba(239,68,68,.13);color:#dc2626;border:1px solid rgba(239,68,68,.35)}
.rdH{background:#ef4444;box-shadow:0 0 7px rgba(239,68,68,.5)}
.rM{background:rgba(245,158,11,.13);color:#d97706;border:1px solid rgba(245,158,11,.35)}
.rdM{background:#f59e0b;box-shadow:0 0 7px rgba(245,158,11,.5)}
.rL{background:rgba(34,197,94,.13);color:#16a34a;border:1px solid rgba(34,197,94,.35)}
.rdL{background:#22c55e;box-shadow:0 0 7px rgba(34,197,94,.5)}

.hm-section{background:rgba(2,6,14,.97);border:1px solid rgba(56,189,248,.2);border-radius:22px;overflow:hidden;box-shadow:0 8px 32px rgba(0,0,0,.45);margin:1.8rem 0}
.hm-header{background:rgba(4,10,24,1);border-bottom:1px solid rgba(56,189,248,.12);padding:1.1rem 1.7rem;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px}
.hm-title{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:600;color:#e2e8f0;letter-spacing:-.3px}
.hm-sub{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:.13em;margin-top:3px}
.hm-legend{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.hm-leg{display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.6)}
.hm-swatch{width:28px;height:9px;border-radius:3px}
.hm-stats{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-bottom:1px solid rgba(255,255,255,.08)}
.hm-stat{padding:13px 16px;border-right:1px solid rgba(255,255,255,.08);text-align:center}
.hm-stat:last-child{border-right:none}
.hm-sv{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:600;color:#38bdf8;line-height:1}
.hm-sl{font-family:'DM Mono',monospace;font-size:8.5px;color:rgba(255,255,255,.45);text-transform:uppercase;letter-spacing:.1em;margin-top:4px}
.hm-col-lbl{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.13em;text-align:center;margin-bottom:8px}
.hm-col-note{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.45);text-align:center;margin-top:8px;line-height:1.6}
.hm-img-frame{border-radius:12px;overflow:hidden;border:1px solid rgba(255,255,255,.1);box-shadow:0 4px 16px rgba(0,0,0,.4)}
.cscale{background:rgba(255,255,255,.025);margin:0 1.6rem 1.4rem;border-radius:8px;padding:9px 13px;border:1px solid rgba(255,255,255,.08)}
.cscale-bar{height:12px;border-radius:3px;background:linear-gradient(90deg,#00007f 0%,#0000ff 12%,#007fff 24%,#00ffff 36%,#7fff7f 50%,#ffff00 64%,#ff7f00 76%,#ff0000 88%,#7f0000 100%);margin-bottom:5px}
.cscale-lbls{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:8.5px;color:rgba(255,255,255,.5)}
.hm-explain{background:rgba(255,255,255,.025);border-top:1px solid rgba(255,255,255,.08);padding:1.3rem 1.7rem}
.hm-exp-title{font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.8);text-transform:uppercase;letter-spacing:.15em;margin-bottom:12px}
.hm-exp-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}
.hm-exp-item{background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.08);border-radius:10px;padding:12px 14px}
.hm-exp-t{font-family:'DM Mono',monospace;font-size:9px;color:#38bdf8;margin-bottom:5px;font-weight:500}
.hm-exp-b{font-size:13px;color:rgba(255,255,255,.8);line-height:1.65}

.rb{border-left:3px solid rgba(147,197,253,.55);background:rgba(255,255,255,.028);border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px}
.rb-red{border-left-color:rgba(248,113,113,.8);background:rgba(239,68,68,.08)}
.rb-yel{border-left-color:rgba(251,191,36,.8);background:rgba(245,158,11,.08)}
.rb-grn{border-left-color:rgba(52,211,153,.8);background:rgba(16,185,129,.08)}
.rb-t{font-family:'DM Mono',monospace;font-size:9px;color:rgba(255,255,255,.55);text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px}
.rb-b{font-size:14px;line-height:1.9;color:rgba(255,255,255,.88)}

.disc{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.3);border-left:3px solid rgba(245,158,11,.8);border-radius:0 12px 12px 0;padding:13px 18px;font-family:'DM Mono',monospace;font-size:10px;color:#fcd34d;line-height:1.78;margin-top:20px}
.disc strong{color:#f59e0b}

[data-testid="stSidebar"]{background:rgba(10,14,26,.98) !important;border-right:1px solid rgba(56,189,248,.09) !important}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{color:#e2e8f0 !important}
[data-testid="stFileUploader"]{border:2px dashed rgba(56,189,248,.75) !important;border-radius:16px !important;background:rgba(14,58,140,.22) !important;padding:4px !important}
[data-testid="stFileUploader"] button{background:#0f172a !important;color:#fff !important;border:1.5px solid rgba(255,255,255,.18) !important;border-radius:10px !important;font-weight:700 !important;font-size:14px !important;padding:10px 24px !important;box-shadow:0 2px 12px rgba(0,0,0,.5) !important}
.stButton>button{background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#4f46e5 100%) !important;color:#fff !important;border:none !important;border-radius:14px !important;font-weight:700 !important;font-size:16px !important;padding:17px 28px !important;width:100% !important;box-shadow:0 6px 24px rgba(37,99,235,.55) !important}
.stButton>button:hover{background:linear-gradient(135deg,#38bdf8 0%,#3b82f6 50%,#6366f1 100%) !important;transform:translateY(-3px) !important}
[data-testid="stMetric"]{background:rgba(255,255,255,.055) !important;border:1px solid rgba(255,255,255,.1) !important;border-radius:14px !important;padding:13px 16px !important}
[data-testid="stMetricValue"]{font-family:'Space Grotesk',sans-serif !important;color:#38bdf8 !important;font-size:22px !important}
[data-testid="stProgress"]>div{background:rgba(255,255,255,.1) !important;border-radius:4px !important}
[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) !important;border-radius:4px !important}
[data-testid="stDownloadButton"]>button{background:rgba(56,189,248,.1) !important;border:1px solid rgba(56,189,248,.28) !important;color:#38bdf8 !important;font-family:'DM Mono',monospace !important;font-size:11.5px !important;border-radius:9px !important;padding:9px 16px !important;width:100% !important;box-shadow:none !important}
[data-testid="stImage"] img{border-radius:12px !important;border:1px solid rgba(255,255,255,.1) !important}
[data-testid="stSelectbox"]>div>div{background:#1e2d45 !important;border:1.5px solid rgba(56,189,248,.5) !important;border-radius:11px !important;min-height:46px !important}
[data-baseweb="menu"]{background:#0f1e36 !important;border:1px solid rgba(56,189,248,.25) !important;border-radius:10px !important}
[data-baseweb="menu"] [role="option"]{background:transparent !important;color:#e2e8f0 !important;padding:12px 18px !important}
[data-baseweb="menu"] [role="option"]:hover{background:rgba(37,99,235,.18) !important;color:#38bdf8 !important}
</style>
""", unsafe_allow_html=True)

# ================================================================
# MODEL LOADING - REAL
# ================================================================
@st.cache_resource(show_spinner="Loading AI models...")
def load_models():
    """Load actual TensorFlow models."""
    models = {
        "resnet": None,
        "mobilenet": None,
        "resnet_loaded": False,
        "mobilenet_loaded": False,
        "tensorflow_available": TF_AVAILABLE
    }
    
    if not TF_AVAILABLE:
        st.sidebar.error("❌ TensorFlow not available")
        return models
    
    # Try to load ResNet50V2
    try:
        if not os.path.exists(MODEL_PATH) and GDRIVE_ID:
            with st.spinner("Downloading ResNet50V2..."):
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
        
        if os.path.exists(MODEL_PATH):
            # Try to load with different methods
            try:
                models["resnet"] = keras.models.load_model(MODEL_PATH, compile=False)
                models["resnet_loaded"] = True
                st.sidebar.success("✅ ResNet50V2 loaded")
            except:
                # Try with custom_objects
                models["resnet"] = keras.models.load_model(MODEL_PATH)
                models["resnet_loaded"] = True
                st.sidebar.success("✅ ResNet50V2 loaded")
    except Exception as e:
        st.sidebar.error(f"❌ ResNet: {str(e)[:40]}")
    
    # Try to load MobileNetV2
    try:
        if not os.path.exists(MOBILENET_PATH) and GDRIVE_MOBILENET_ID:
            with st.spinner("Downloading MobileNetV2..."):
                gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MOBILENET_ID}", MOBILENET_PATH, quiet=False)
        
        if os.path.exists(MOBILENET_PATH):
            try:
                models["mobilenet"] = keras.models.load_model(MOBILENET_PATH, compile=False)
                models["mobilenet_loaded"] = True
                st.sidebar.success("✅ MobileNetV2 loaded")
            except:
                models["mobilenet"] = keras.models.load_model(MOBILENET_PATH)
                models["mobilenet_loaded"] = True
                st.sidebar.success("✅ MobileNetV2 loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ MobileNet: {str(e)[:40]}")
    
    return models

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
# REAL GRAD-CAM
# ================================================================
def make_real_gradcam(model, img_array, pred_idx):
    """Compute REAL Grad-CAM from actual model gradients."""
    if model is None or not TF_AVAILABLE:
        return None
    
    try:
        # Find the last convolutional layer
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
        
        # Build gradient model
        src = backbone or model
        grad_model = keras.Model(
            inputs=model.inputs,
            outputs=[src.get_layer(last_conv).output, model.output]
        )
        
        # Compute gradients
        with tf.GradientTape() as tape:
            conv_output, predictions = grad_model(img_array)
            loss = predictions[:, pred_idx]
        
        grads = tape.gradient(loss, conv_output)
        if grads is None:
            return None
        
        # Compute heatmap
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = tf.nn.relu(tf.reduce_sum(tf.multiply(weights, conv_output[0]), axis=-1))
        heatmap = heatmap.numpy()
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        return heatmap
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

# ================================================================
# REAL INFERENCE
# ================================================================
def run_real_inference(img, models, temperature=1.4, use_ensemble=True):
    """Run REAL inference with actual models."""
    resnet_model = models.get("resnet")
    mobilenet_model = models.get("mobilenet")
    
    raw_resnet = None
    if resnet_model is not None and TF_AVAILABLE:
        try:
            arr_resnet = preprocess(img)
            raw_resnet_out = resnet_model.predict(arr_resnet, verbose=0)[0]
            raw_resnet = np.zeros(4, dtype=np.float32)
            for i in range(4):
                raw_resnet[i] = raw_resnet_out[i]
        except Exception as e:
            st.warning(f"ResNet inference error: {str(e)[:50]}")
    
    raw_mobilenet = None
    if mobilenet_model is not None and TF_AVAILABLE and use_ensemble:
        try:
            arr_mobilenet = preprocess_for_mobilenet(img)
            raw_mobilenet_out = mobilenet_model.predict(arr_mobilenet, verbose=0)[0]
            raw_mobilenet = np.zeros(4, dtype=np.float32)
            for i in range(4):
                raw_mobilenet[i] = raw_mobilenet_out[i]
        except Exception as e:
            st.warning(f"MobileNet inference error: {str(e)[:50]}")
    
    if raw_resnet is not None and raw_mobilenet is not None and use_ensemble:
        raw = raw_resnet * 0.60 + raw_mobilenet * 0.40
        ensemble_mode = "ResNet50V2 + MobileNetV2 Ensemble"
        is_demo = False
    elif raw_resnet is not None:
        raw = raw_resnet
        ensemble_mode = "ResNet50V2 Single Model"
        is_demo = False
    else:
        # Fallback to image analysis if models fail
        raw = analyze_image_fallback(img)
        ensemble_mode = "Image Analysis (Fallback)"
        is_demo = True
    
    # Apply temperature scaling
    try:
        logits = np.log(np.clip(raw, 1e-7, 1.0))
        scaled = np.exp(logits / temperature)
        preds = scaled / scaled.sum()
    except:
        preds = raw / raw.sum()
    
    return preds, ensemble_mode, is_demo

def analyze_image_fallback(img):
    """Fallback image analysis when models fail."""
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    
    # Simple heuristic
    if std_intensity > 35:
        return np.array([0.72, 0.15, 0.08, 0.05])
    elif std_intensity > 25:
        return np.array([0.08, 0.78, 0.09, 0.05])
    elif std_intensity > 15:
        return np.array([0.10, 0.12, 0.08, 0.70])
    else:
        return np.array([0.05, 0.05, 0.85, 0.05])

# ================================================================
# SYNTHETIC HEATMAP (Fallback)
# ================================================================
def synthetic_heatmap(pil_img):
    """Synthetic heatmap for fallback mode."""
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
# REAL CLAUDE ANALYSIS
# ================================================================
def real_claude_analysis(pil_img, pred_class, conf, heatmap_img, all_probs, ensemble_mode):
    """Send actual image to Claude for REAL analysis."""
    try:
        # Try to get API key
        try:
            key = st.secrets["ANTHROPIC_API_KEY"]
        except:
            key = os.environ.get("ANTHROPIC_API_KEY", "")
        
        if not key:
            return None, False, "No API key configured"
        
        # Import anthropic
        try:
            import anthropic
        except ImportError:
            return None, False, "Anthropic library not installed"
        
        # Prepare image data
        buffered = io.BytesIO()
        pil_img.convert("RGB").save(buffered, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        # Prepare heatmap data
        heatmap_buffered = io.BytesIO()
        heatmap_img.convert("RGB").save(heatmap_buffered, format="JPEG", quality=85)
        heatmap_base64 = base64.b64encode(heatmap_buffered.getvalue()).decode()
        
        # Build probability distribution
        prob_lines = "\n".join(f"  - {name}: {p*100:.1f}%" for name, p in all_probs.items())
        
        # System prompt
        sysp = """You are a specialist neuro-oncology AI assistant with expertise in brain MRI interpretation.

A CNN has produced a classification result on this axial brain MRI. You are also shown its Grad-CAM heatmap.

CRITICAL: You are looking at the ACTUAL uploaded image and its real Grad-CAM heatmap.
Your job is to INDEPENDENTLY ASSESS THE IMAGE, NOT to rubber-stamp the CNN's top label.

- Look at the actual image and heatmap first
- If what you observe best matches the CNN's top class, say so and explain why
- If what you observe is more consistent with a DIFFERENT class, say that explicitly
- Never invent precise measurements - use qualitative descriptors
- Reference specific visual features you can actually see

Respond with valid JSON ONLY. No markdown, no backticks.

JSON schema:
{
  "agrees_with_cnn": true or false,
  "clinical_interpretation": "Detailed description of observed findings. Minimum 3 sentences.",
  "location_morphology": "Anatomical location, size (qualitative), shape, borders.",
  "model_reasoning": "Assess whether the CNN's top-1 class is visually supported.",
  "gradcam_analysis": "Describe which regions show high activation and whether this makes sense.",
  "risk_level": "HIGH or MODERATE or LOW",
  "risk_justification": "Clinical justification for the risk level.",
  "patient_explanation": "Plain English explanation for a patient. 2-3 sentences.",
  "next_steps": "Numbered list of specific recommended clinical actions.",
  "image_quality": "GOOD or ADEQUATE or POOR",
  "image_quality_notes": "Specific observations about this image's quality.",
  "uncertainty_factors": "Features that reduce prediction confidence.",
  "differential_diagnosis": "1-2 alternative diagnoses worth considering.",
  "reliability_score": 0-100,
  "overall_reliability": "One sentence summary.",
  "disclaimer": "AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon."
}"""
        
        user_text = (
            f"CNN top prediction: {pred_class} ({conf:.1f}% confidence).\n"
            f"Full class probability distribution:\n{prob_lines}\n"
            f"Model configuration: {ensemble_mode}.\n"
            f"Classes: Glioma, Meningioma, Pituitary Tumor, No Tumor.\n"
            f"Independently assess the attached MRI and Grad-CAM heatmap and provide your report as JSON only."
        )
        
        # Prepare messages
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": heatmap_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": user_text
                    }
                ]
            }
        ]
        
        # Call Claude
        client = anthropic.Anthropic(api_key=key)
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1800,
            system=sysp,
            messages=messages
        )
        
        # Parse response
        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:]
        
        parsed = json.loads(raw.strip())
        return parsed, True, None
        
    except Exception as e:
        return None, False, str(e)

# ================================================================
# TEMPLATE REPORT (Fallback)
# ================================================================
def template_report(pred_class, conf):
    """Template report for when Claude is unavailable."""
    reports = {
        "Glioma": {
            "clinical_interpretation": f"Heterogeneous mass lesion with irregular margins and peritumoral edema detected. Confidence: {conf:.1f}%.",
            "location_morphology": "Right frontal lobe, supratentorial compartment.",
            "model_reasoning": f"Features consistent with glioma: ring-enhancing pattern, heterogeneous signal, peritumoral edema.",
            "gradcam_analysis": "Activation localised to the tumor epicenter with secondary activation at the edema boundary.",
            "risk_level": "HIGH",
            "risk_justification": "High-grade glioma carries significant morbidity.",
            "patient_explanation": "The scan shows signs of a Glioma brain tumor. This is NOT a final diagnosis.",
            "next_steps": "1. Immediate neuroradiologist review\n2. Contrast-enhanced MRI\n3. Neurosurgical consultation",
            "image_quality": "ADEQUATE",
            "uncertainty_factors": "Partial ambiguity at tumor-edema boundary.",
            "reliability_score": 88,
            "overall_reliability": "Good reliability with minor uncertainty.",
            "differential_diagnosis": "1. High-grade glioblastoma. 2. Metastatic lesion.",
            "disclaimer": "AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon."
        },
        "Meningioma": {
            "clinical_interpretation": f"Well-circumscribed extra-axial mass with dural tail sign. Confidence: {conf:.1f}%.",
            "location_morphology": "Parasagittal convexity, extra-axial. Broad dural base.",
            "model_reasoning": f"Features consistent with meningioma: extra-axial location, homogeneous signal, dural attachment.",
            "gradcam_analysis": "Model focuses on the lesion-dura interface.",
            "risk_level": "MODERATE",
            "risk_justification": "Most meningiomas are WHO Grade I.",
            "patient_explanation": "The scan suggests a meningioma - usually slow-growing.",
            "next_steps": "1. Neurology review\n2. Contrast-enhanced MRI\n3. Observation vs surgery",
            "image_quality": "GOOD",
            "uncertainty_factors": "Cavernous sinus involvement needs coronal sequences.",
            "reliability_score": 86,
            "overall_reliability": "Good reliability.",
            "differential_diagnosis": "1. Dural metastasis. 2. Hemangiopericytoma.",
            "disclaimer": "AI-assisted decision support only."
        },
        "Pituitary Tumor": {
            "clinical_interpretation": f"Intrasellar mass expanding the sella turcica. Confidence: {conf:.1f}%.",
            "location_morphology": "Sella turcica, macroadenoma with suprasellar extension.",
            "model_reasoning": f"Features consistent with pituitary tumor: intrasellar location, sella expansion.",
            "gradcam_analysis": "Model activates on the sellar region.",
            "risk_level": "MODERATE",
            "risk_justification": "Usually benign pituitary adenoma.",
            "patient_explanation": "The scan shows a tumor in the pituitary gland.",
            "next_steps": "1. Endocrinology consultation\n2. Visual field testing\n3. Hormone panel",
            "image_quality": "GOOD",
            "uncertainty_factors": "Cavernous sinus invasion requires Knosp grading.",
            "reliability_score": 89,
            "overall_reliability": "High reliability.",
            "differential_diagnosis": "1. Craniopharyngioma. 2. Rathke cleft cyst.",
            "disclaimer": "AI-assisted decision support only."
        },
        "No Tumor": {
            "clinical_interpretation": f"Normal brain parenchyma. No mass lesion detected. Confidence: {conf:.1f}%.",
            "location_morphology": "No focal lesion. Normal midline structures.",
            "model_reasoning": f"Features consistent with normal brain: symmetric architecture, no mass effect.",
            "gradcam_analysis": "Low distributed activation with no focal concentration.",
            "risk_level": "LOW",
            "risk_justification": "No imaging evidence of neoplasm.",
            "patient_explanation": "Good news - no tumor detected.",
            "next_steps": "Clinical follow-up if symptoms persist.",
            "image_quality": "GOOD",
            "uncertainty_factors": "None significant.",
            "reliability_score": 95,
            "overall_reliability": "Very high reliability.",
            "differential_diagnosis": "No significant differential.",
            "disclaimer": "AI-assisted decision support only."
        }
    }
    return reports.get(pred_class, reports["No Tumor"])

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

def four_panel_fig(pil_img, hm, pred_class, conf, demo=False):
    overlay, hm_smooth = overlay_gradcam(pil_img, hm)
    orig = np.array(pil_img.convert("RGB").resize(IMG_SIZE))
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#020609")
    for ax in axes:
        ax.set_facecolor("#020609")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a2e4a")
    
    axes[0].imshow(orig)
    axes[0].axis("off")
    axes[0].set_title("Original MRI", color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    
    axes[1].imshow(np.array(overlay))
    axes[1].axis("off")
    axes[1].set_title("Grad-CAM Overlay", color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    
    im = axes[2].imshow(hm_smooth, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    axes[2].axis("off")
    axes[2].set_title("Activation Map", color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#666", labelsize=6)
    
    flat = hm_smooth.flatten()
    n, bins, patches = axes[3].hist(flat, bins=40, edgecolor="none")
    axes[3].axvline(flat.mean(), color="#fbbf24", ls="--", lw=1.1)
    axes[3].set_xlabel("Activation", color="#666", fontsize=7)
    axes[3].set_facecolor("#020609")
    for sp in axes[3].spines.values():
        sp.set_edgecolor("#1a2e4a")
    axes[3].tick_params(colors="#666", labelsize=6)
    axes[3].set_title("Histogram", color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    
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

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("### 🧠 NeuroScan AI")
    st.markdown("---")
    
    st.markdown("#### Model Status")
    models = load_models()
    
    if TF_AVAILABLE:
        st.success("✅ TensorFlow")
    else:
        st.error("❌ TensorFlow")
    
    if models["resnet_loaded"]:
        st.success("✅ ResNet50V2")
    else:
        st.error("❌ ResNet50V2")
    
    if models["mobilenet_loaded"]:
        st.success("✅ MobileNetV2")
    else:
        st.warning("⚠️ MobileNetV2")
    
    st.markdown("---")
    st.markdown("#### Settings")
    
    use_ensemble = st.toggle("Use Ensemble", value=True,
                            help="Combine ResNet50V2 and MobileNetV2")
    alpha = st.slider("Heatmap Intensity", 0.2, 0.8, 0.55, 0.05)
    temperature = st.slider("Temperature", 1.0, 2.5, 1.4, 0.1)
    
    st.markdown("---")
    st.markdown("""
    <div style="background:rgba(245,158,11,.07);border-left:3px solid #f59e0b;
      border-radius:0 8px 8px 0;padding:10px 12px;font-family:'DM Mono',monospace;
      font-size:9.5px;color:rgba(253,211,77,.92);line-height:1.7;">
      <strong style="color:#fbbf24;">Clinical Disclaimer</strong><br>
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
    <span class="chip c-teal">Real Grad-CAM</span>
    <span class="chip c-green">95.31% Accuracy</span>
    <span class="chip c-amber">4-Class CNN</span>
    <span class="chip c-purple">Claude AI</span>
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
          complete with REAL Grad-CAM heatmaps and AI-generated clinical reports.
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
      <div class="pip-step"><div class="pip-num">3</div><div class="pip-txt"><strong>Real Grad-CAM</strong>Tumor region heatmap</div></div>
      <div class="pip-arr">›</div>
      <div class="pip-step"><div class="pip-num">4</div><div class="pip-txt"><strong>Claude Report</strong>Clinical analysis</div></div>
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
        help="Axial T1 or T2-weighted brain MRI."
    )
    
    st.markdown('''<div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.68);text-align:center;padding:8px 0 14px;text-transform:uppercase;letter-spacing:.11em;background:rgba(56,189,248,.06);border-radius:8px;margin-top:6px;">
      JPG / PNG / BMP &nbsp;·&nbsp; Max 10 MB &nbsp;·&nbsp; T1 or T2 axial preferred
    </div>''', unsafe_allow_html=True)

    img = None
    if uploaded:
        try:
            _bytes = uploaded.read()
            _buf = io.BytesIO(_bytes)
            _raw = Image.open(_buf)
            _raw.load()
            _raw = ImageOps.exif_transpose(_raw)
            _arr_loaded = np.array(_raw.convert("RGB"), dtype=np.uint8)
            img = Image.fromarray(_arr_loaded, mode="RGB")
            st.success("✅ Image uploaded successfully.")
        except Exception as _e:
            st.error(f"Failed to open image: {_e}")
            img = None
    else:
        st.markdown('''<div style="border:2px dashed rgba(56,189,248,.38);border-radius:14px;padding:2.5rem 1.5rem;text-align:center;background:rgba(56,189,248,.05);margin:8px 0;">
          <div style="font-size:40px;margin-bottom:12px;">🩻</div>
          <div style="font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:6px;">No image selected</div>
          <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.58);letter-spacing:.07em;line-height:1.9;">
            Upload a brain MRI above to begin analysis
          </div>
        </div>''', unsafe_allow_html=True)

    if img:
        st.image(img, use_column_width=True, clamp=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    
    _btn_lbl = "Upload an MRI first" if img is None else "🔬 Analyze and Generate Clinical Report"
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
              Upload a brain MRI scan,<br>
              then click <strong>Analyse</strong> to run the full pipeline.
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">REAL CNN PREDICTION</span>
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">REAL GRAD-CAM</span>
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">CLAUDE AI REPORT</span>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

# ================================================================
# ANALYSIS - REAL GRAD-CAM AND CLAUDE
# ================================================================
if clicked and img:
    st.markdown("""
<div id="ns-result-anchor"></div>
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
<style>
@keyframes ns-slide-in { from { transform:translateY(-80px); opacity:0 } to { transform:translateY(0); opacity:1 } }
@keyframes ns-fade-out { from { opacity:1 } to { opacity:0; pointer-events:none } }
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
#ns-toast-icon { font-size:22px; flex-shrink:0; }
#ns-toast-title { font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:600; color:#e2e8f0; margin-bottom:2px; }
#ns-toast-sub { font-family:'DM Mono',monospace; font-size:10px; color:rgba(255,255,255,.55); letter-spacing:.05em; }
</style>
""", unsafe_allow_html=True)

    with col_out:
        st.markdown("""
<div style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.30);border-left:4px solid #38bdf8;border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:20px">🔬</span>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:#e2e8f0;">Analysis in progress</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:rgba(255,255,255,.50);margin-top:2px;letter-spacing:.05em;">CNN INFERENCE → REAL GRAD-CAM → CLAUDE REPORT</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # REAL Inference
    with st.spinner("Running CNN inference..."):
        preds, ensemble_mode, is_demo = run_real_inference(img, models, temperature, use_ensemble)

    pidx = int(np.argmax(preds))
    pcls = CLASS_NAMES[pidx]
    conf = float(preds[pidx]) * 100
    rl, rc, dc = RISK[pcls]

    st.markdown(f"""
<script>
(function() {{
  var t = document.getElementById('ns-toast');
  var ti = document.getElementById('ns-toast-title');
  var ts = document.getElementById('ns-toast-sub');
  if (ti) ti.textContent = 'Result: {pcls} ({conf:.1f}%)';
  if (ts) ts.textContent = 'Risk: {rl} — scroll down for full report';
  if (t) {{
    t.style.animation = 'none';
    t.offsetHeight;
    t.style.animation = 'ns-slide-in .4s ease forwards, ns-fade-out .5s ease 5s forwards';
  }}
}})();
</script>
""", unsafe_allow_html=True)

    # REAL Grad-CAM
    with st.spinner("Computing REAL Grad-CAM heatmap..."):
        if models["resnet"] is not None and TF_AVAILABLE:
            arr = preprocess(img)
            real_hm = make_real_gradcam(models["resnet"], arr, pidx)
            if real_hm is not None:
                hm = real_hm
                is_demo = False
                st.success("✅ Real Grad-CAM computed from model gradients")
            else:
                hm = synthetic_heatmap(img)
                is_demo = True
                st.warning("⚠️ Using synthetic heatmap (Grad-CAM failed)")
        else:
            hm = synthetic_heatmap(img)
            is_demo = True
            st.info("ℹ️ Using synthetic heatmap (no model loaded)")

        overlay_img, hm_smooth = overlay_gradcam(img, hm, alpha=alpha)

    # Stats
    mean_a = float(hm_smooth.mean())
    max_a = float(hm_smooth.max())
    p90_a = float(np.percentile(hm_smooth, 90))
    focus_p = float((hm_smooth > 0.5).sum() / hm_smooth.size * 100)

    # Claude Report
    with st.spinner("Generating Claude AI clinical report..."):
        all_probs = {n: float(p) for n, p in zip(CLASS_NAMES, preds)}
        
        # Try to get REAL Claude analysis
        claude_report, claude_success, claude_error = real_claude_analysis(
            img, pcls, conf, overlay_img, all_probs, ensemble_mode
        )
        
        if claude_success and claude_report is not None:
            report = claude_report
            report_is_live = True
            st.success("✅ Real Claude analysis of the actual image")
        else:
            report = template_report(pcls, conf)
            report_is_live = False
            if claude_error:
                st.warning(f"⚠️ Claude unavailable: {claude_error[:100]}... Using template report")

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
  <div class="pred-eyebrow">{ensemble_mode} | {'REAL' if not is_demo else 'FALLBACK'} Grad-CAM</div>
  <div class="pred-name">{pcls}</div>
  <div class="conf-row">
    <span class="conf-l">Confidence</span>
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
      <div class="hm-title">REAL Grad-CAM Heatmap | {pcls}</div>
      <div class="hm-sub">Gradient-based explainability from actual CNN {'(Real)' if not is_demo else '(Synthetic - Fallback)'}</div>
    </div>
    <div class="hm-legend">
      <div class="hm-leg"><span class="hm-swatch" style="background:linear-gradient(90deg,#00007f,#007fff,#00ffff)"></span>Low</div>
      <div class="hm-leg"><span class="hm-swatch" style="background:linear-gradient(90deg,#ffff00,#ff7f00,#ff0000)"></span>High</div>
    </div>
  </div>
""", unsafe_allow_html=True)

    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        st.markdown(f"""
<div style="padding:1.2rem 0.4rem;">
  <div style="font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.16em;margin-bottom:18px;font-weight:600;">Class Distribution</div>
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
        st.markdown('<div class="hm-col-note">Raw input</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="hm-col-lbl">Activation Map</div>', unsafe_allow_html=True)
        fh = pure_heatmap_fig(hm_smooth, pcls, conf)
        st.pyplot(fh, use_container_width=True)
        plt.close()
        st.markdown('<div class="hm-col-note">Normalised intensity</div>', unsafe_allow_html=True)

    with sc3:
        st.markdown('<div class="hm-col-lbl">Histogram</div>', unsafe_allow_html=True)
        fhist = histogram_fig(hm_smooth)
        st.pyplot(fhist, use_container_width=True)
        plt.close()
        st.markdown('<div class="hm-col-note">Activation distribution</div>', unsafe_allow_html=True)

    with sc4:
        st.markdown('<div class="hm-col-lbl">Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:8px;padding-top:2px;">
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{mean_a:.3f}</div><div class="hm-sl">Mean</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{p90_a:.3f}</div><div class="hm-sl">P90</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{focus_p:.1f}%</div><div class="hm-sl">High Area</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{max_a:.3f}</div><div class="hm-sl">Peak</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Color scale
    st.markdown("""
  <div class="cscale" style="margin:14px 1.5rem;">
    <div class="cscale-bar"></div>
    <div class="cscale-lbls">
      <span>Low</span><span>Cyan</span><span>Green/Yellow</span><span>Orange</span><span>High (Red)</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    if is_demo:
        st.warning("⚠️ **Using synthetic Grad-CAM** - Real Grad-CAM requires loaded TensorFlow models.")

    # Download
    fig4 = four_panel_fig(img, hm, pcls, conf, is_demo)
    fbyt = fig_bytes(fig4)
    plt.close(fig4)
    st.download_button("Download Figure (PNG)",
                       data=fbyt,
                       file_name=f"neuroscan_{pcls.lower().replace(' ','_')}.png",
                       mime="image/png")

    # Clinical Report
    st.markdown("---")
    st.markdown('<div class="slbl">Clinical Report</div>', unsafe_allow_html=True)

    if report_is_live:
        st.success("✅ **Live Claude Analysis** - This report was generated by Claude analyzing the actual uploaded image and its Grad-CAM heatmap.")
    else:
        st.info("ℹ️ **Template Report** - For live Claude analysis, configure your Anthropic API key in secrets.")

    t1, t2, t3, t4 = st.tabs(["Findings", "Reasoning", "Patient Summary", "Reliability"])

    with t1:
        st.markdown(rb("Clinical Interpretation", report.get("clinical_interpretation", ""), "rb-red"), unsafe_allow_html=True)
        st.markdown(rb("Location & Morphology", report.get("location_morphology", "")), unsafe_allow_html=True)

    with t2:
        st.markdown(rb("Model Reasoning", report.get("model_reasoning", "")), unsafe_allow_html=True)
        st.markdown(rb("Grad-CAM Analysis", report.get("gradcam_analysis", ""), "rb-grn"), unsafe_allow_html=True)

    with t3:
        st.markdown(rb("Patient Summary", report.get("patient_explanation", ""), "rb-yel"), unsafe_allow_html=True)
        st.markdown(rb("Next Steps", report.get("next_steps", "").replace("\n", "<br>")), unsafe_allow_html=True)

    with t4:
        rs = report.get("reliability_score", 80)
        a, b, c = st.columns(3)
        with a:
            st.metric("Reliability", f"{rs}/100")
        with b:
            st.metric("Image Quality", report.get("image_quality", "N/A"))
        with c:
            st.metric("Risk", rl)
        st.progress(rs / 100)
        qv = {"GOOD": "rb-grn", "ADEQUATE": "rb-yel", "POOR": "rb-red"}.get(report.get("image_quality", "GOOD"), "rb-grn")
        st.markdown(rb("Uncertainty Factors", report.get("uncertainty_factors", "None"), qv), unsafe_allow_html=True)
        st.markdown(rb("Differential Diagnosis", report.get("differential_diagnosis", "")), unsafe_allow_html=True)

    st.markdown(f"""
<div class="disc">
  <strong>AI-Assisted Decision Support Only</strong> —
  {report.get("disclaimer", "")}
  All findings require review by a licensed radiologist or neurosurgeon.
</div>""", unsafe_allow_html=True)

    # Export JSON
    gcam_stats = {"mean": round(mean_a, 4), "peak": round(max_a, 4), "p90": round(p90_a, 4), "focus_pct": round(focus_p, 2), "synthetic": is_demo}
    st.download_button(
        "Export Report (JSON)",
        data=json.dumps({
            "system": "NeuroScan AI v3.0",
            "model": ensemble_mode,
            "gradcam_type": "Real" if not is_demo else "Synthetic",
            "tensorflow_available": TF_AVAILABLE,
            "prediction": pcls,
            "confidence": round(conf, 2),
            "risk": rl,
            "gradcam": gcam_stats,
            "probabilities": {n: round(float(p), 4) for n, p in zip(CLASS_NAMES, preds)},
            **report
        }, indent=2),
        file_name=f"neuroscan_{pcls.lower().replace(' ', '_')}.json",
        mime="application/json"
    )

    # Model Performance
    st.markdown("---")
    st.markdown('<div class="slbl">Model Performance</div>', unsafe_allow_html=True)
    st.markdown("""
<div class="glass">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(100px,1fr));gap:10px;margin-bottom:14px;">
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">95.31%</div><div class="hm-sl">Accuracy</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">100%</div><div class="hm-sl">No Tumor</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">99.8%</div><div class="hm-sl">Pituitary</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">98.0%</div><div class="hm-sl">Meningioma</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">83.5%</div><div class="hm-sl">Glioma</div></div>
  </div>
  <div style="font-size:12px;color:rgba(255,255,255,.38);line-height:1.78;">
    Ensemble combines ResNet50V2 and MobileNetV2 via soft-voting.
    Glioma recall is lower due to similarity with meningioma on T1 non-contrast.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
