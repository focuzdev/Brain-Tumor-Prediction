"""
NeuroScan AI - Brain Tumor MRI Classification + Grad-CAM Heatmap
================================================================
Model    : ResNet50V2 + MobileNetV2 Ensemble | 4 classes | 95.31% accuracy
XAI      : Grad-CAM via TensorFlow GradientTape
Status   : FULLY FIXED - Class mapping auto-detection + Ensemble forced
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
import traceback
from datetime import datetime

# Try importing anthropic (optional)
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

# Theme
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
_dk = (st.session_state.theme == "dark")

# ================================================================
# CSS STYLING (abbreviated for readability - keep full CSS from previous version)
# ================================================================
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

*,*::before,*::after{{box-sizing:border-box}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;-webkit-font-smoothing:antialiased}}

.stApp{{
  background:{"#0a0e1a" if _dk else "#f0f4fa"} !important;
  background-image:
    radial-gradient(ellipse 90% 55% at 10% -5%,{"rgba(14,58,150,.35)" if _dk else "rgba(186,220,255,.45)"} 0%,transparent 45%),
    radial-gradient(ellipse 70% 45% at 95% 105%,{"rgba(8,100,160,.20)" if _dk else "rgba(170,220,245,.30)"} 0%,transparent 45%) !important;
  color:{"#e2e8f0" if _dk else "#0a1628"} !important;
  min-height:100vh;
}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding:0 !important;max-width:100% !important}}

/* Top Navigation */
.topnav{{
  position:sticky;top:0;z-index:200;
  background:{"rgba(10,14,26,.95)" if _dk else "rgba(255,255,255,.97)"};
  backdrop-filter:blur(24px) saturate(180%);
  border-bottom:1px solid {"rgba(56,189,248,.15)" if _dk else "#b0cfe0"};
  padding:.8rem 2.4rem;
  display:flex;align-items:center;justify-content:space-between;gap:1rem;
}}
.nav-brand{{display:flex;align-items:center;gap:13px}}
.nav-logo{{width:38px;height:38px;border-radius:10px;font-size:18px;background:linear-gradient(135deg,#1e40af,#0e7490);display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px rgba(56,189,248,.40);flex-shrink:0}}
.nav-name{{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:{"#e2e8f0" if _dk else "#0a1628"};letter-spacing:-.3px}}
.nav-name span{{color:#38bdf8}}
.nav-tagline{{font-family:'DM Mono',monospace;font-size:8px;color:{"rgba(255,255,255,.45)" if _dk else "#4a6580"};letter-spacing:.15em;text-transform:uppercase;margin-top:1px}}
.nav-right{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.chip{{font-family:'DM Mono',monospace;font-size:9px;font-weight:500;padding:4px 10px;border-radius:20px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}}
.c-blue{{background:rgba(59,130,246,.14);color:#93c5fd;border:1px solid rgba(59,130,246,.28)}}
.c-teal{{background:rgba(20,184,166,.14);color:#5eead4;border:1px solid rgba(20,184,166,.28)}}
.c-green{{background:rgba(34,197,94,.14);color:#86efac;border:1px solid rgba(34,197,94,.28)}}
.c-amber{{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.28)}}
.c-purple{{background:rgba(139,92,246,.14);color:#c4b5fd;border:1px solid rgba(139,92,246,.28)}}
.c-red{{background:rgba(239,68,68,.14);color:#fca5a5;border:1px solid rgba(239,68,68,.28)}}

.theme-toggle{{
  width:34px;height:34px;border-radius:50%;
  background:{"rgba(28,38,68,.85)" if _dk else "#daeeff"};
  border:1px solid {"rgba(56,189,248,.50)" if _dk else "#4a9ab8"};
  color:{"#7dd3fc" if _dk else "#084e65"};
  font-size:16px;line-height:1;cursor:pointer;flex-shrink:0;
  display:inline-flex;align-items:center;justify-content:center;
  box-shadow:0 2px 10px rgba(0,0,0,.15);
  transition:background .18s,transform .18s !important;
}}
.theme-toggle:hover{{
  background:{"rgba(56,189,248,.25)" if _dk else "#b8dff0"};
  transform:scale(1.12) rotate(14deg);
}}

/* Hero */
.hero{{position:relative;overflow:hidden;padding:3rem 0 2.5rem;background:linear-gradient(130deg,{"#040c1c" if _dk else "#daeeff"} 0%,{"#071630" if _dk else "#c4e0f8"} 55%,{"#040c1c" if _dk else "#daeeff"} 100%);border-bottom:1px solid {"rgba(56,189,248,.09)" if _dk else "#a8cfe6"}}}
.hero-inner{{position:relative;z-index:1;width:100%;padding:0 2.8rem;box-sizing:border-box}}
.hero-top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}}
.hero-h1{{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.9rem,3.5vw,3rem);font-weight:700;color:{"#e2e8f0" if _dk else "#0a1628"};letter-spacing:-.7px;line-height:1.13;margin-bottom:.5rem}}
.hero-h1 .grad{{background:linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.hero-desc{{font-size:15px;color:{"rgba(255,255,255,.70)" if _dk else "#2d4a6b"};line-height:1.74;max-width:530px}}
.hero-stats{{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end}}
.hs{{text-align:right}}
.hs-n{{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;background:linear-gradient(92deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}}
.hs-l{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.12em;margin-top:3px}}
.hero-div{{height:1px;margin:1.2rem 0;background:linear-gradient(90deg,rgba(56,189,248,.30),rgba(129,140,248,.18),transparent)}}
.pipeline{{display:flex;align-items:center;gap:0;flex-wrap:wrap}}
.pip-step{{display:flex;align-items:center;gap:9px}}
.pip-num{{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0891b2);font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px rgba(56,189,248,.32);flex-shrink:0}}
.pip-txt{{font-size:11.5px;color:{"rgba(255,255,255,.65)" if _dk else "#2d4a6b"};line-height:1.38}}
.pip-txt strong{{color:{"rgba(255,255,255,.90)" if _dk else "#0a1628"};display:block;font-size:11px}}
.pip-arr{{color:{"rgba(56,189,248,.40)" if _dk else "#4a8fa8"};font-size:20px;padding:0 10px}}

/* Wrapper */
.wrap{{width:100%;padding:2rem 2.8rem 4rem;box-sizing:border-box}}
.glass{{background:{"rgba(255,255,255,.030)" if _dk else "rgba(255,255,255,.92)"};border:1px solid {"rgba(255,255,255,.075)" if _dk else "#b8d4e8"};border-radius:20px;padding:1.8rem 2rem;backdrop-filter:blur(12px);box-shadow:0 8px 40px {"rgba(0,0,0,.35)" if _dk else "rgba(10,22,80,.08)"}}}
.slbl{{font-family:'DM Mono',monospace;font-size:11px;color:{"rgba(56,189,248,.80)" if _dk else "#0e6b8a"};text-transform:uppercase;letter-spacing:.17em;margin-bottom:12px;display:flex;align-items:center;gap:8px}}
.slbl::after{{content:'';flex:1;height:1px;background:{"rgba(56,189,248,.20)" if _dk else "#9ecadb"}}}

/* Prediction Card */
.pred-card{{background:linear-gradient(135deg,{"rgba(14,30,70,.82)" if _dk else "#daeeff"},{"rgba(8,20,48,.92)" if _dk else "#cce5f8"});border:1px solid {"rgba(56,189,248,.22)" if _dk else "#7ab8d4"};border-radius:18px;padding:1.5rem 1.6rem;margin-bottom:14px;position:relative;overflow:hidden;box-shadow:0 12px 40px {"rgba(0,0,0,.25)" if _dk else "rgba(10,22,80,.10)"}}}
.pred-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8,#818cf8)}}
.pred-eyebrow{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(56,189,248,.80)" if _dk else "#0e6b8a"};text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px}}
.pred-name{{font-family:'Space Grotesk',sans-serif;font-size:clamp(30px,4vw,44px);font-weight:700;color:{"#f8fafc" if _dk else "#0a1628"};letter-spacing:-1px;line-height:1.04;margin-bottom:15px}}
.conf-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}}
.conf-l{{font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"}}}
.conf-v{{font-family:'Space Grotesk',sans-serif;font-size:14px;color:#38bdf8;font-weight:600}}
.conf-track{{background:{"rgba(255,255,255,.10)" if _dk else "#b8d8ec"};border-radius:8px;height:6px;overflow:hidden;margin-bottom:15px}}
.conf-fill{{height:100%;border-radius:8px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8)}}
.risk-chip{{display:inline-flex;align-items:center;gap:7px;padding:6px 15px;border-radius:20px;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}}
.rdot{{width:6px;height:6px;border-radius:50%}}
.rH{{background:rgba(239,68,68,.13);color:#dc2626;border:1px solid rgba(239,68,68,.35)}}
.rdH{{background:#ef4444;box-shadow:0 0 7px rgba(239,68,68,.5)}}
.rM{{background:rgba(245,158,11,.13);color:#d97706;border:1px solid rgba(245,158,11,.35)}}
.rdM{{background:#f59e0b;box-shadow:0 0 7px rgba(245,158,11,.5)}}
.rL{{background:rgba(34,197,94,.13);color:#16a34a;border:1px solid rgba(34,197,94,.35)}}
.rdL{{background:#22c55e;box-shadow:0 0 7px rgba(34,197,94,.5)}}

/* Heatmap Section */
.hm-section{{background:{"rgba(2,6,14,.97)" if _dk else "#f4faff"};border:1px solid {"rgba(56,189,248,.20)" if _dk else "#a0cce0"};border-radius:22px;overflow:hidden;box-shadow:0 8px 32px {"rgba(0,0,0,.45)" if _dk else "rgba(10,22,80,.08)"};margin:1.8rem 0}}
.hm-header{{background:{"rgba(4,10,24,1)" if _dk else "#daeeff"};border-bottom:1px solid {"rgba(56,189,248,.12)" if _dk else "#a8cfe0"};padding:1.1rem 1.7rem;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px}}
.hm-title{{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:600;color:{"#e2e8f0" if _dk else "#0a1628"};letter-spacing:-.3px}}
.hm-sub{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.45)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.13em;margin-top:3px}}
.hm-legend{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.hm-leg{{display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.60)" if _dk else "#1a3550"}}}
.hm-swatch{{width:28px;height:9px;border-radius:3px}}
.hm-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-bottom:1px solid {"rgba(255,255,255,.08)" if _dk else "#b8d8ec"}}}
.hm-stat{{padding:13px 16px;border-right:1px solid {"rgba(255,255,255,.08)" if _dk else "#b8d8ec"};text-align:center}}
.hm-stat:last-child{{border-right:none}}
.hm-sv{{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:600;color:#38bdf8;line-height:1}}
.hm-sl{{font-family:'DM Mono',monospace;font-size:8.5px;color:{"rgba(255,255,255,.45)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.1em;margin-top:4px}}
.hm-col-lbl{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#1a3550"};text-transform:uppercase;letter-spacing:.13em;text-align:center;margin-bottom:8px}}
.hm-col-note{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.45)" if _dk else "#2d4a6b"};text-align:center;margin-top:8px;line-height:1.6}}
.hm-img-frame{{border-radius:12px;overflow:hidden;border:1px solid {"rgba(255,255,255,.10)" if _dk else "#a8cfe0"};box-shadow:0 4px 16px {"rgba(0,0,0,.40)" if _dk else "rgba(10,22,80,.10)"}}}
.cscale{{background:{"rgba(255,255,255,.025)" if _dk else "#ffffff"};margin:0 1.6rem 1.4rem;border-radius:8px;padding:9px 13px;border:1px solid {"rgba(255,255,255,.08)" if _dk else "#b8d8ec"}}}
.cscale-bar{{height:12px;border-radius:3px;background:linear-gradient(90deg,#00007f 0%,#0000ff 12%,#007fff 24%,#00ffff 36%,#7fff7f 50%,#ffff00 64%,#ff7f00 76%,#ff0000 88%,#7f0000 100%);margin-bottom:5px}}
.cscale-lbls{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:8.5px;color:{"rgba(255,255,255,.50)" if _dk else "#2d4a6b"}}}
.hm-explain{{background:{"rgba(255,255,255,.025)" if _dk else "#ffffff"};border-top:1px solid {"rgba(255,255,255,.08)" if _dk else "#a8cfe0"};padding:1.3rem 1.7rem}}
.hm-exp-title{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(56,189,248,.80)" if _dk else "#0e6b8a"};text-transform:uppercase;letter-spacing:.15em;margin-bottom:12px}}
.hm-exp-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}
.hm-exp-item{{background:{"rgba(255,255,255,.025)" if _dk else "#ffffff"};border:1px solid {"rgba(255,255,255,.08)" if _dk else "#a8cfe0"};border-radius:10px;padding:12px 14px}}
.hm-exp-t{{font-family:'DM Mono',monospace;font-size:9px;color:#38bdf8;margin-bottom:5px;font-weight:500}}
.hm-exp-b{{font-size:13px;color:{"rgba(255,255,255,.80)" if _dk else "#0a1628"};line-height:1.65}}

/* Report Blocks */
.rb{{border-left:3px solid rgba(147,197,253,.55);background:{"rgba(255,255,255,.028)" if _dk else "#ffffff"};border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px}}
.rb-red{{border-left-color:rgba(248,113,113,.80);background:{"rgba(239,68,68,.08)" if _dk else "#fde8e8"}}}
.rb-yel{{border-left-color:rgba(251,191,36,.80);background:{"rgba(245,158,11,.08)" if _dk else "#fef3cd"}}}
.rb-grn{{border-left-color:rgba(52,211,153,.80);background:{"rgba(16,185,129,.08)" if _dk else "#d8f5e8"}}}
.rb-t{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px}}
.rb-b{{font-size:14px;line-height:1.90;color:{"rgba(255,255,255,.88)" if _dk else "#0a1628"}}}

/* Disclaimer */
.disc{{background:{"rgba(245,158,11,.08)" if _dk else "#fef6d8"};border:1px solid {"rgba(245,158,11,.30)" if _dk else "#d4a017"};border-left:3px solid rgba(245,158,11,.80);border-radius:0 12px 12px 0;padding:13px 18px;font-family:'DM Mono',monospace;font-size:10px;color:{"#fcd34d" if _dk else "#7a4800"};line-height:1.78;margin-top:20px}}
.disc strong{{color:#f59e0b}}

/* Streamlit Overrides */
[data-testid="stSidebar"]{{background:{"rgba(10,14,26,.98)" if _dk else "#f0f8ff"} !important;border-right:1px solid {"rgba(56,189,248,.09)" if _dk else "#a8cfe0"} !important}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{{color:{"#e2e8f0" if _dk else "#0a1628"} !important}}
.stTabs [data-baseweb="tab-list"]{{background:{"rgba(255,255,255,.050)" if _dk else "#ddeef8"} !important;border-radius:11px !important;padding:3px !important;gap:2px !important;border:1px solid {"rgba(255,255,255,.090)" if _dk else "#a8cfe0"} !important}}
.stTabs [data-baseweb="tab"]{{border-radius:8px !important;font-family:'DM Mono',monospace !important;font-size:10.5px !important;color:{"rgba(255,255,255,.60)" if _dk else "#1e3a55"} !important;padding:8px 15px !important}}
.stTabs [aria-selected="true"]{{background:rgba(56,189,248,.15) !important;color:#38bdf8 !important;box-shadow:none !important}}
[data-testid="stFileUploader"]{{border:2px dashed {"rgba(56,189,248,.75)" if _dk else "#0e7490"} !important;border-radius:16px !important;background:{"rgba(14,58,140,.22)" if _dk else "rgba(12,74,110,.06)"} !important;padding:4px !important}}
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small{{color:{"#7dd3fc" if _dk else "#0a4a60"} !important;font-family:'DM Mono',monospace !important;font-size:12px !important}}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button{{background:#0f172a !important;color:#ffffff !important;border:1.5px solid rgba(255,255,255,.18) !important;border-radius:10px !important;font-family:'Space Grotesk',sans-serif !important;font-weight:700 !important;font-size:14px !important;padding:10px 24px !important;opacity:1 !important;min-width:110px !important;box-shadow:0 2px 12px rgba(0,0,0,.50) !important;transition:background .15s ease,box-shadow .15s ease,transform .15s ease !important;-webkit-text-fill-color:#ffffff !important}}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover,
[data-testid="stFileUploaderDropzone"] button:hover{{background:#1e293b !important;transform:translateY(-2px) !important;box-shadow:0 8px 24px rgba(0,0,0,.60) !important;border-color:rgba(255,255,255,.30) !important}}
.stButton > button,
button[kind="primary"],
button[kind="secondary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"]{{background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#4f46e5 100%) !important;color:#ffffff !important;-webkit-text-fill-color:#ffffff !important;border:none !important;border-radius:14px !important;font-family:'Space Grotesk',sans-serif !important;font-weight:700 !important;font-size:16px !important;padding:17px 28px !important;width:100% !important;transition:all .22s ease !important;letter-spacing:.03em !important;box-shadow:0 6px 24px rgba(37,99,235,.55),0 2px 8px rgba(0,0,0,.20) !important;text-shadow:0 1px 2px rgba(0,0,0,.20) !important}}
.stButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover{{background:linear-gradient(135deg,#38bdf8 0%,#3b82f6 50%,#6366f1 100%) !important;transform:translateY(-3px) !important;box-shadow:0 12px 36px rgba(37,99,235,.65),0 4px 12px rgba(0,0,0,.25) !important}}
[data-testid="stMetric"]{{background:{"rgba(255,255,255,.055)" if _dk else "#ffffff"} !important;border:1px solid {"rgba(255,255,255,.10)" if _dk else "#a8cfe0"} !important;border-radius:14px !important;padding:13px 16px !important}}
[data-testid="stMetricLabel"]{{color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"} !important;font-size:11px !important}}
[data-testid="stMetricValue"]{{font-family:'Space Grotesk',sans-serif !important;color:#38bdf8 !important;font-size:22px !important}}
[data-testid="stProgress"] > div{{background:{"rgba(255,255,255,.10)" if _dk else "#b8d8ec"} !important;border-radius:4px !important}}
[data-testid="stProgress"] > div > div{{background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) !important;border-radius:4px !important}}
[data-testid="stDownloadButton"] > button{{background:{"rgba(56,189,248,.10)" if _dk else "#daeeff"} !important;border:1px solid {"rgba(56,189,248,.28)" if _dk else "#4a9ab8"} !important;color:{"#38bdf8" if _dk else "#084e65"} !important;font-family:'DM Mono',monospace !important;font-size:11.5px !important;border-radius:9px !important;padding:9px 16px !important;width:100% !important;box-shadow:none !important}}
code{{font-family:'DM Mono',monospace !important;font-size:11px !important;background:{"rgba(255,255,255,.08)" if _dk else "#e8f4fb"} !important;color:{"#7dd3fc" if _dk else "#0a4570"} !important;border:1px solid {"rgba(255,255,255,.10)" if _dk else "#a0c8dc"} !important;border-radius:4px !important}}
hr{{border-color:{"rgba(255,255,255,.10)" if _dk else "#c2d8e8"} !important;margin:1.8rem 0 !important}}
[data-testid="stImage"] img{{border-radius:12px !important;border:1px solid {"rgba(255,255,255,.10)" if _dk else "#a8cfe0"} !important;display:block !important}}
div[data-testid="stSuccess"]{{background:{"rgba(34,197,94,.10)" if _dk else "#d4f5e2"} !important;border-color:{"rgba(34,197,94,.30)" if _dk else "#2e9e5e"} !important;color:{"#86efac" if _dk else "#0a4020"} !important}}
div[data-testid="stError"]{{background:{"rgba(239,68,68,.10)" if _dk else "#fde0e0"} !important;border-color:{"rgba(239,68,68,.30)" if _dk else "#d04040"} !important;color:{"#fca5a5" if _dk else "#5e0a0a"} !important}}
div[data-testid="stWarning"]{{background:{"rgba(245,158,11,.10)" if _dk else "#fef3cd"} !important;border-color:{"rgba(245,158,11,.30)" if _dk else "#c08000"} !important;color:{"#fcd34d" if _dk else "#6a3800"} !important}}
div[data-testid="stInfo"]{{background:{"rgba(56,189,248,.09)" if _dk else "#d8eefb"} !important;border-color:{"rgba(56,189,248,.28)" if _dk else "#2878a8"} !important;color:{"#7dd3fc" if _dk else "#083858"} !important}}
[data-testid="stSelectbox"] > div > div{{background:{"#1e2d45" if _dk else "#ffffff"} !important;border:1.5px solid {"rgba(56,189,248,.50)" if _dk else "#0e7490"} !important;border-radius:11px !important;min-height:46px !important}}
[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child{{color:{"#f1f5f9" if _dk else "#0a1628"} !important;-webkit-text-fill-color:{"#f1f5f9" if _dk else "#0a1628"} !important;font-family:'Space Grotesk',sans-serif !important;font-size:14px !important;font-weight:500 !important}}
[data-testid="stSelectbox"] svg{{fill:{"#38bdf8" if _dk else "#0e7490"} !important;color:{"#38bdf8" if _dk else "#0e7490"} !important;flex-shrink:0 !important}}
[data-baseweb="popover"] ul,
[data-baseweb="menu"]{{background:{"#0f1e36" if _dk else "#ffffff"} !important;border:1px solid {"rgba(56,189,248,.25)" if _dk else "#b0d4e4"} !important;border-radius:10px !important;overflow:hidden !important;box-shadow:0 8px 32px rgba(0,0,0,.35) !important}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"]{{background:transparent !important;color:
