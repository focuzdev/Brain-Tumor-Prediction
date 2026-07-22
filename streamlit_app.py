"""
NeuroScan AI - Brain Tumor MRI Classification + Grad-CAM Heatmap
================================================================
Model    : ResNet50V2  |  4 classes  |  95.31% ensemble accuracy
XAI      : Grad-CAM via TensorFlow GradientTape (NO VLM needed)
Heatmap  : ALWAYS renders - real CNN gradients when model loaded,
           synthetic (MRI-derived) in demo mode. Toggle has been
           removed so the visualization is never accidentally hidden.
"""

import streamlit as st
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image
import io, base64, os, json
import anthropic
import gdown

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
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Theme stored in session_state. Default = light.
# Toggle calls st.rerun() — Python re-injects correct CSS every time.
if "theme" not in st.session_state:
    st.session_state.theme = "light"

_dk = (st.session_state.theme == "dark")  # True = dark, False = light

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');
*,*::before,*::after{{box-sizing:border-box}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;-webkit-font-smoothing:antialiased}}

/* ── ALL COLORS ARE PYTHON VARIABLES — correct on every render ── */

/* app background */
.stApp{{
  background:{"#04090f" if _dk else "#eef4fb"} !important;
  background-image:
    radial-gradient(ellipse 90% 55% at 10% -5%,{"rgba(14,58,150,.45)" if _dk else "rgba(186,220,255,.55)"} 0%,transparent 45%),
    radial-gradient(ellipse 70% 45% at 95% 105%,{"rgba(8,100,160,.25)" if _dk else "rgba(170,220,245,.38)"} 0%,transparent 45%),
    radial-gradient(ellipse 50% 70% at 50% 50%,{"rgba(5,20,50,.40)" if _dk else "rgba(205,228,250,.30)"} 0%,transparent 70%) !important;
  color:{"#e2e8f0" if _dk else "#0a1628"} !important;
  min-height:100vh;
}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding:0 !important;max-width:100% !important}}

/* nav */
.topnav{{
  position:sticky;top:0;z-index:200;
  background:{"rgba(4,9,15,.92)" if _dk else "rgba(255,255,255,.97)"};
  backdrop-filter:blur(24px) saturate(180%);
  -webkit-backdrop-filter:blur(24px) saturate(180%);
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
.c-blue  {{background:rgba(59,130,246,.14);color:#93c5fd;border:1px solid rgba(59,130,246,.28)}}
.c-teal  {{background:rgba(20,184,166,.14);color:#5eead4;border:1px solid rgba(20,184,166,.28)}}
.c-green {{background:rgba(34,197,94,.14);color:#86efac;border:1px solid rgba(34,197,94,.28)}}
.c-amber {{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.28)}}
.c-purple{{background:rgba(139,92,246,.14);color:#c4b5fd;border:1px solid rgba(139,92,246,.28)}}

/* hero */
.hero{{position:relative;overflow:hidden;padding:3.5rem 0 3rem;background:linear-gradient(130deg,{"#040c1c" if _dk else "#daeeff"} 0%,{"#071630" if _dk else "#c4e0f8"} 55%,{"#040c1c" if _dk else "#daeeff"} 100%);border-bottom:1px solid {"rgba(56,189,248,.09)" if _dk else "#a8cfe6"}}}
.hero::before{{content:'';position:absolute;inset:0;pointer-events:none;background:radial-gradient(ellipse 60% 80% at 82% 45%,rgba(56,189,248,.07) 0%,transparent 55%),radial-gradient(ellipse 40% 60% at 18% 75%,rgba(99,102,241,.06) 0%,transparent 50%)}}
.hero-inner{{position:relative;z-index:1;width:100%;padding:0 2.8rem;box-sizing:border-box}}
.hero-top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}}
.hero-h1{{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.9rem,3.5vw,3rem);font-weight:700;color:{"#e2e8f0" if _dk else "#0a1628"};letter-spacing:-.7px;line-height:1.13;margin-bottom:.5rem}}
.hero-h1 .grad{{background:linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.hero-desc{{font-size:15px;color:{"rgba(255,255,255,.70)" if _dk else "#2d4a6b"};line-height:1.74;max-width:530px}}
.hero-stats{{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end}}
.hs{{text-align:right}}
.hs-n{{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;background:linear-gradient(92deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}}
.hs-l{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.12em;margin-top:3px}}
.hero-div{{height:1px;margin:1.4rem 0;background:linear-gradient(90deg,rgba(56,189,248,.30),rgba(129,140,248,.18),transparent)}}
.pipeline{{display:flex;align-items:center;gap:0;flex-wrap:wrap}}
.pip-step{{display:flex;align-items:center;gap:9px}}
.pip-num{{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0891b2);font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px rgba(56,189,248,.32);flex-shrink:0}}
.pip-txt{{font-size:11.5px;color:{"rgba(255,255,255,.65)" if _dk else "#2d4a6b"};line-height:1.38}}
.pip-txt strong{{color:{"rgba(255,255,255,.90)" if _dk else "#0a1628"};display:block;font-size:11px}}
.pip-arr{{color:{"rgba(56,189,248,.40)" if _dk else "#4a8fa8"};font-size:20px;padding:0 10px}}

/* wrap + glass + section label */
.wrap{{width:100%;padding:2.4rem 2.8rem 5rem;box-sizing:border-box}}
.glass{{background:{"rgba(255,255,255,.030)" if _dk else "rgba(255,255,255,.92)"};border:1px solid {"rgba(255,255,255,.075)" if _dk else "#b8d4e8"};border-radius:20px;padding:2rem 2.2rem;backdrop-filter:blur(12px);box-shadow:0 8px 40px {"rgba(0,0,0,.35)" if _dk else "rgba(10,22,80,.08)"}}}
.slbl{{font-family:'DM Mono',monospace;font-size:11px;color:{"rgba(56,189,248,.80)" if _dk else "#0e6b8a"};text-transform:uppercase;letter-spacing:.17em;margin-bottom:12px;display:flex;align-items:center;gap:8px}}
.slbl::after{{content:'';flex:1;height:1px;background:{"rgba(56,189,248,.20)" if _dk else "#9ecadb"}}}

/* prediction card */
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

/* heatmap section */
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
.hm-grid{{padding:1.4rem 1.6rem}}
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

/* report blocks */
.rb    {{border-left:3px solid rgba(147,197,253,.55);background:{"rgba(255,255,255,.028)" if _dk else "#ffffff"};border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px}}
.rb-red{{border-left-color:rgba(248,113,113,.80);background:{"rgba(239,68,68,.08)" if _dk else "#fde8e8"}}}
.rb-yel{{border-left-color:rgba(251,191,36,.80);background:{"rgba(245,158,11,.08)" if _dk else "#fef3cd"}}}
.rb-grn{{border-left-color:rgba(52,211,153,.80);background:{"rgba(16,185,129,.08)" if _dk else "#d8f5e8"}}}
.rb-t{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px}}
.rb-b{{font-size:14px;line-height:1.90;color:{"rgba(255,255,255,.88)" if _dk else "#0a1628"}}}

/* disclaimer */
.disc{{background:{"rgba(245,158,11,.08)" if _dk else "#fef6d8"};border:1px solid {"rgba(245,158,11,.30)" if _dk else "#d4a017"};border-left:3px solid rgba(245,158,11,.80);border-radius:0 12px 12px 0;padding:13px 18px;font-family:'DM Mono',monospace;font-size:10px;color:{"#fcd34d" if _dk else "#7a4800"};line-height:1.78;margin-top:20px}}
.disc strong{{color:#f59e0b}}

/* streamlit overrides */
[data-testid="stSidebar"]{{background:{"rgba(4,9,15,.98)" if _dk else "#f0f8ff"} !important;border-right:1px solid {"rgba(56,189,248,.09)" if _dk else "#a8cfe0"} !important}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{{color:{"#e2e8f0" if _dk else "#0a1628"} !important}}
.stTabs [data-baseweb="tab-list"]{{background:{"rgba(255,255,255,.050)" if _dk else "#ddeef8"} !important;border-radius:11px !important;padding:3px !important;gap:2px !important;border:1px solid {"rgba(255,255,255,.090)" if _dk else "#a8cfe0"} !important}}
.stTabs [data-baseweb="tab"]{{border-radius:8px !important;font-family:'DM Mono',monospace !important;font-size:10.5px !important;color:{"rgba(255,255,255,.60)" if _dk else "#1e3a55"} !important;padding:8px 15px !important}}
.stTabs [aria-selected="true"]{{background:rgba(56,189,248,.15) !important;color:#38bdf8 !important;box-shadow:none !important}}
[data-testid="stFileUploader"]{{
  border:2px dashed {"rgba(56,189,248,.75)" if _dk else "#0e7490"} !important;
  border-radius:16px !important;
  background:{"rgba(14,58,140,.22)" if _dk else "rgba(12,74,110,.06)"} !important;
  padding:4px !important;
}}
/* Only style the hint text — NOT the button children (avoids icon duplication) */
[data-testid="stFileUploaderDropzoneInstructions"] p,
[data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploaderDropzoneInstructions"] small{{
  color:{"#7dd3fc" if _dk else "#0a4a60"} !important;
  font-family:'DM Mono',monospace !important;
  font-size:12px !important;
}}
/* Upload button — clean, high-contrast, no text corruption */
/* Upload button — multiple selectors for Streamlit version compatibility */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"],
[data-testid="stFileUploaderDropzone"] button{{
  background:#0f172a !important;
  color:#ffffff !important;
  border:1.5px solid rgba(255,255,255,.18) !important;
  border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-weight:700 !important;
  font-size:14px !important;
  padding:10px 24px !important;
  opacity:1 !important;
  min-width:110px !important;
  box-shadow:0 2px 12px rgba(0,0,0,.50) !important;
  transition:background .15s ease,box-shadow .15s ease,transform .15s ease !important;
  -webkit-text-fill-color:#ffffff !important;
}}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]:hover,
[data-testid="stFileUploaderDropzone"] button:hover{{
  background:#1e293b !important;
  transform:translateY(-2px) !important;
  box-shadow:0 8px 24px rgba(0,0,0,.60) !important;
  border-color:rgba(255,255,255,.30) !important;
}}
.stButton > button,
button[kind="primary"],
button[kind="secondary"],
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"]{{
  background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#4f46e5 100%) !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  border:none !important;
  border-radius:14px !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-weight:700 !important;
  font-size:16px !important;
  padding:17px 28px !important;
  width:100% !important;
  transition:all .22s ease !important;
  letter-spacing:.03em !important;
  box-shadow:0 6px 24px rgba(37,99,235,.55),0 2px 8px rgba(0,0,0,.20) !important;
  text-shadow:0 1px 2px rgba(0,0,0,.20) !important;
}}
.stButton > button:hover,
button[kind="primary"]:hover,
button[kind="secondary"]:hover{{
  background:linear-gradient(135deg,#38bdf8 0%,#3b82f6 50%,#6366f1 100%) !important;
  transform:translateY(-3px) !important;
  box-shadow:0 12px 36px rgba(37,99,235,.65),0 4px 12px rgba(0,0,0,.25) !important;
}}
.stButton > button:active{{
  transform:translateY(-1px) !important;
  box-shadow:0 4px 14px rgba(37,99,235,.40) !important;
}}
/* ── SELECTBOX — surgical targeting, no SVG interference ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div{{
  background:{"#1e2d45" if _dk else "#ffffff"} !important;
  border:1.5px solid {"rgba(56,189,248,.50)" if _dk else "#0e7490"} !important;
  border-radius:11px !important;
  min-height:46px !important;
}}
/* The visible text value or placeholder */
[data-testid="stSelectbox"] [data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div:first-child,
[data-testid="stSelectbox"] [data-baseweb="select"] [role="combobox"]{{
  color:{"#f1f5f9" if _dk else "#0a1628"} !important;
  -webkit-text-fill-color:{"#f1f5f9" if _dk else "#0a1628"} !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-size:14px !important;
  font-weight:500 !important;
}}
/* Arrow icon only */
[data-testid="stSelectbox"] svg{{
  fill:{"#38bdf8" if _dk else "#0e7490"} !important;
  color:{"#38bdf8" if _dk else "#0e7490"} !important;
  flex-shrink:0 !important;
}}
/* Dropdown popover panel */
[data-baseweb="popover"] ul,
[data-baseweb="menu"]{{
  background:{"#0f1e36" if _dk else "#ffffff"} !important;
  border:1px solid {"rgba(56,189,248,.25)" if _dk else "#b0d4e4"} !important;
  border-radius:10px !important;
  overflow:hidden !important;
  box-shadow:0 8px 32px rgba(0,0,0,.35) !important;
}}
[data-baseweb="popover"] [role="option"],
[data-baseweb="menu"] [role="option"]{{
  background:transparent !important;
  color:{"#e2e8f0" if _dk else "#0a1628"} !important;
  -webkit-text-fill-color:{"#e2e8f0" if _dk else "#0a1628"} !important;
  font-family:'Space Grotesk',sans-serif !important;
  font-size:14px !important;
  font-weight:500 !important;
  padding:12px 18px !important;
  border-bottom:1px solid {"rgba(255,255,255,.05)" if _dk else "#e8f4fb"} !important;
  transition:background .15s !important;
}}
[data-baseweb="menu"] [role="option"]:hover{{
  background:{"rgba(37,99,235,.18)" if _dk else "rgba(14,116,144,.10)"} !important;
  color:{"#38bdf8" if _dk else "#0e6b8a"} !important;
  -webkit-text-fill-color:{"#38bdf8" if _dk else "#0e6b8a"} !important;
}}
[data-baseweb="menu"] [aria-selected="true"]{{
  background:{"rgba(37,99,235,.25)" if _dk else "rgba(14,116,144,.14)"} !important;
  color:{"#38bdf8" if _dk else "#0e6b8a"} !important;
  -webkit-text-fill-color:{"#38bdf8" if _dk else "#0e6b8a"} !important;
  font-weight:600 !important;
}}
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
div[data-testid="stError"]  {{background:{"rgba(239,68,68,.10)" if _dk else "#fde0e0"} !important;border-color:{"rgba(239,68,68,.30)" if _dk else "#d04040"} !important;color:{"#fca5a5" if _dk else "#5e0a0a"} !important}}
div[data-testid="stWarning"]{{background:{"rgba(245,158,11,.10)" if _dk else "#fef3cd"} !important;border-color:{"rgba(245,158,11,.30)" if _dk else "#c08000"} !important;color:{"#fcd34d" if _dk else "#6a3800"} !important}}
div[data-testid="stInfo"]   {{background:{"rgba(56,189,248,.09)" if _dk else "#d8eefb"} !important;border-color:{"rgba(56,189,248,.28)" if _dk else "#2878a8"} !important;color:{"#7dd3fc" if _dk else "#083858"} !important}}

/* theme toggle button */
.theme-toggle{{
  width:34px;height:34px;border-radius:50%;
  background:{"rgba(28,38,68,.85)" if _dk else "#daeeff"};
  border:1px solid {"rgba(56,189,248,.50)" if _dk else "#4a9ab8"};
  color:{"#7dd3fc" if _dk else "#084e65"};
  font-size:16px;line-height:1;
  display:inline-flex;align-items:center;justify-content:center;
  cursor:pointer;flex-shrink:0;
  box-shadow:0 2px 10px rgba(0,0,0,.15);
  -webkit-appearance:none;appearance:none;outline:none;
  text-decoration:none;user-select:none;
  transition:background .18s,transform .18s !important;
}}
.theme-toggle:hover{{
  background:{"rgba(56,189,248,.25)" if _dk else "#b8dff0"};
  transform:scale(1.12) rotate(14deg);
}}
</style>
""", unsafe_allow_html=True)

# ================================================================
# CONSTANTS
# ================================================================
CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
CLASS_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7"]
IMG_SIZE     = (224, 224)
MODEL_PATH   = "brain_tumor_model.h5"
SAMPLE_DIR   = "samples"
GDRIVE_ID    = os.environ.get("GDRIVE_FILE_ID", "")

# MobileNetV2 companion model for the true soft-vote ensemble. This was
# missing entirely before: the repo only ships brain_tumor_model.h5
# (ResNet50V2), so "ensemble" mode never actually ran in production even
# though the UI and 95.31% accuracy badge advertised it everywhere.
MOBILENET_PATH = "mobilenet_model.h5"
GDRIVE_MOBILENET_ID = os.environ.get("GDRIVE_MOBILENET_FILE_ID", "")

RISK = {
    "Glioma":          ("HIGH",     "rH", "rdH"),
    "Meningioma":      ("MODERATE", "rM", "rdM"),
    "Pituitary Tumor": ("MODERATE", "rM", "rdM"),
    "No Tumor":        ("LOW",      "rL", "rdL"),
}
SAMPLES = {
    "Select a sample image": None,
    "Glioma":          "glioma.jpg",
    "Meningioma":      "meningioma.jpg",
    "Pituitary Tumor": "pituitary.jpg",
    "No Tumor":        "no_tumor.jpg",
}

# ================================================================
# MODEL
# ================================================================
@st.cache_resource(show_spinner="Loading CNN model...")
def load_model():
    if not TF_AVAILABLE: return None
    if not os.path.exists(MODEL_PATH) and GDRIVE_ID:
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_ID}", MODEL_PATH, quiet=False)
    return keras.models.load_model(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

@st.cache_resource(show_spinner="Loading MobileNetV2 ensemble member...")
def load_mobilenet():
    """
    Loads the second ensemble member. Mirrors load_model()'s download
    fallback so the real ensemble can actually run in deployment instead
    of silently degrading to solo ResNet50V2. Cached with st.cache_resource
    so it isn't re-read from disk on every single prediction (the previous
    version called keras.models.load_model(mob_path) fresh inside the
    per-image inference path, which is both slow and unnecessary).
    """
    if not TF_AVAILABLE: return None
    if not os.path.exists(MOBILENET_PATH) and GDRIVE_MOBILENET_ID:
        gdown.download(f"https://drive.google.com/uc?id={GDRIVE_MOBILENET_ID}",
                        MOBILENET_PATH, quiet=False)
    return keras.models.load_model(MOBILENET_PATH) if os.path.exists(MOBILENET_PATH) else None

# ================================================================
# PREPROCESSING
# ================================================================
def _resize_and_clean(img):
    """Shared step: resize to model input size and normalize to a clean
    (224, 224, 3) uint8-range float array. NO backbone-specific scaling
    here — that happens separately per model below, since ResNet50V2 and
    MobileNetV2 each define their own canonical preprocess_input."""
    resized = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(resized, dtype=np.float32)
    if arr.ndim == 2:                      # greyscale fallback
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[-1] == 4:               # RGBA — drop alpha
        arr = arr[:, :, :3]
    arr = arr[:IMG_SIZE[0], :IMG_SIZE[1], :3]  # crop if needed
    return arr

def preprocess(img):
    """ResNet50V2 input tensor. Kept as the default single-model path
    (used everywhere the app previously called preprocess())."""
    arr = _resize_and_clean(img)
    arr = resnet_preprocess(arr) if resnet_preprocess is not None else (arr / 127.5 - 1.0)
    return np.expand_dims(arr, 0)

def preprocess_for_mobilenet(img):
    """MobileNetV2 input tensor. NOTE: resnet_v2.preprocess_input and
    mobilenet_v2.preprocess_input currently both use Keras' 'tf' scaling
    mode (x/127.5 - 1), so today these two functions are numerically
    identical. They are kept as SEPARATE calls anyway rather than reusing
    the ResNet-preprocessed array for MobileNetV2's predict() call: that
    equivalence is a coincidence of the current Keras applications
    implementation, not a guarantee, and silently relying on it is a
    trap the moment either preprocessing function changes upstream, or
    if this app is ever pointed at a different MobileNet variant (e.g.
    MobileNetV3, which uses different preprocessing). Always preprocess
    for the model you're actually calling."""
    arr = _resize_and_clean(img)
    arr = mobilenet_preprocess(arr) if mobilenet_preprocess is not None else (arr / 127.5 - 1.0)
    return np.expand_dims(arr, 0)

# ================================================================
# MRI INPUT VALIDATOR  — gates non-MRI images before inference
# ================================================================
def validate_mri(pil_img):
    """
    Hardened 8-signal MRI validator. Returns (is_valid, confidence, reasons).

    The methodology image failure case taught us that greyscale + low-saturation
    is necessary but NOT sufficient — a black-and-white document diagram also
    passes those tests. We need signals that specifically distinguish the
    INTENSITY TOPOLOGY of an MRI (dark surround, bright interior blob) from
    a document (mostly white, with black text lines).

    Signals
    -------
    1. Colour saturation     — HARD GATE: MRI is near-greyscale (sat < 30)
    2. Dark surround         — HARD GATE: MRI must have ≥ 15% near-black pixels
                               (the skull exterior / air void is always black)
    3. White background      — HARD GATE: documents are mostly white (>50% pixels
                               above 230). Real MRI almost never exceeds 35%.
    4. Intensity distribution— MRI has significant dark + mid-grey mass
    5. Bright pixel cap      — MRI rarely has >50% very bright pixels;
                               a white-bg document always does
    6. Edge structure        — text documents have regular, dense, horizontal edges
    7. Aspect ratio          — axial MRI is roughly square (0.65–1.55)
    8. Local contrast        — MRI has smooth gradients; text has sharp black/white
                               transitions giving a high local std-dev pattern
    """
    import numpy as np
    import cv2

    img_rgb  = pil_img.convert("RGB")
    img_gray = pil_img.convert("L")
    w, h     = img_rgb.size
    arr_rgb  = np.array(img_rgb,  dtype=np.float32)
    arr_gray = np.array(img_gray, dtype=np.float32)
    arr_u8   = arr_gray.astype(np.uint8)
    scores   = {}

    # ── Signal 1 (HARD GATE): Colour saturation ──────────────────
    arr_u8_rgb = np.array(img_rgb, dtype=np.uint8)
    hsv      = cv2.cvtColor(arr_u8_rgb, cv2.COLOR_RGB2HSV)
    mean_sat = float(hsv[:,:,1].mean())
    # Also check channel deviation — catches JPEG-compressed MRI with slight noise
    r_c = arr_rgb[:,:,0]; g_c = arr_rgb[:,:,1]; b_c = arr_rgb[:,:,2]
    mean_ch  = (r_c + g_c + b_c) / 3
    ch_dev   = float((np.abs(r_c-mean_ch)+np.abs(g_c-mean_ch)+np.abs(b_c-mean_ch)).mean())
    # Pass if EITHER metric confirms near-greyscale
    # sat < 30 covers true greyscale; ch_dev < 15 covers JPEG-compressed greyscale MRI
    s1 = (mean_sat < 30.0) or (ch_dev < 15.0)
    scores["colour_saturation"] = (s1,
        f"HSV sat {mean_sat:.1f} ch_dev {ch_dev:.1f} — "
        f"{'OK (near-greyscale)' if s1 else 'FAIL: colour image, not MRI'}")

    # ── Signal 2 (HARD GATE): Dark surround ──────────────────────
    dark_ratio = float((arr_gray < 25).sum() / arr_gray.size)
    s2 = dark_ratio >= 0.15
    scores["dark_surround"] = (s2,
        f"Near-black px {dark_ratio*100:.1f}% — "
        f"{'OK (skull void present)' if s2 else 'FAIL: no dark background — likely document/photo'}")

    # ── Signal 3 (HARD GATE): White background rejection ─────────
    # Documents: >50% pixels above 230 (nearly white).
    # Real axial MRI: almost never exceeds 35% bright pixels.
    white_ratio = float((arr_gray > 230).sum() / arr_gray.size)
    s3 = white_ratio < 0.40
    scores["white_background"] = (s3,
        f"Near-white px {white_ratio*100:.1f}% — "
        f"{'OK' if s3 else 'FAIL: predominantly white — likely document, diagram, or screenshot'}")

    # ── Signal 4: Intensity distribution ─────────────────────────
    hist, _ = np.histogram(arr_gray.flatten(), bins=256, range=(0,255))
    dark_m = hist[:40].sum()  / arr_gray.size
    mid_m  = hist[40:200].sum()/ arr_gray.size
    s4 = (dark_m > 0.12) and (mid_m > 0.10)
    scores["intensity_distribution"] = (s4,
        f"Dark {dark_m*100:.0f}% Mid {mid_m*100:.0f}% — "
        f"{'OK' if s4 else 'FAIL: MRI bimodal distribution not found'}")

    # ── Signal 5: Bright pixel cap ───────────────────────────────
    bright_ratio = float((arr_gray > 200).sum() / arr_gray.size)
    s5 = bright_ratio < 0.50
    scores["bright_pixel_cap"] = (s5,
        f"Bright px {bright_ratio*100:.1f}% — "
        f"{'OK' if s5 else 'FAIL: too many bright pixels — white document background'}")

    # ── Signal 6: Edge structure ──────────────────────────────────
    edges = cv2.Canny(arr_u8, 40, 110)
    ed    = float(edges.sum() / 255 / edges.size)
    # Also check horizontal vs vertical edge ratio
    # Text documents have very high horizontal edge ratio (line text)
    sobelx = cv2.Sobel(arr_u8, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(arr_u8, cv2.CV_64F, 0, 1, ksize=3)
    hv_ratio = (np.abs(sobely).sum() / (np.abs(sobelx).sum() + 1e-6))
    # MRI: hv_ratio ~0.8–1.2 (isotropic). Text docs: >1.5 (horizontal dominance)
    s6 = (ed < 0.20) and (hv_ratio < 1.6)
    scores["edge_structure"] = (s6,
        f"Density {ed:.3f} H/V ratio {hv_ratio:.2f} — "
        f"{'OK' if s6 else 'FAIL: document-like edge pattern detected'}")

    # ── Signal 7: Aspect ratio ────────────────────────────────────
    ratio = w / h
    s7 = 0.60 <= ratio <= 1.60
    scores["aspect_ratio"] = (s7,
        f"{w}×{h} ratio {ratio:.2f} — {'OK' if s7 else 'FAIL: unusual aspect ratio for MRI'}")

    # ── Signal 8: Local contrast profile ─────────────────────────
    # Compute local std-dev via a sliding window approximation.
    # MRI has smooth gradual transitions → moderate local std.
    # Text on white paper has very bimodal local std (near 0 in white regions,
    # very high at text edges) → high mean of local variance.
    kernel = np.ones((8,8), np.float32) / 64
    local_mean = cv2.filter2D(arr_gray, -1, kernel)
    local_sq   = cv2.filter2D(arr_gray**2, -1, kernel)
    local_var  = np.clip(local_sq - local_mean**2, 0, None)
    local_std  = np.sqrt(local_var)
    mean_lstd  = float(local_std.mean())
    # MRI: typically 15–55. Pure white/black docs: 0–15 in flat areas but
    # the global mean is pulled down. Diagrams: 8–25.
    # Key insight: MRI mean local std is almost always > 20
    s8 = mean_lstd > 12.0  # lowered for JPEG-compressed MRI; still rejects flat docs
    scores["local_contrast"] = (s8,
        f"Mean local σ {mean_lstd:.1f} — "
        f"{'OK (smooth MRI gradients)' if s8 else 'FAIL: flat regions suggest document, not MRI'}")

    # ── Decision logic ────────────────────────────────────────────
    # Three HARD GATES must ALL pass (signals 1, 2, 3).
    # Then at least 3 of the remaining 5 soft signals must pass.
    hard_ok  = s1 and s2 and s3
    soft_pass = sum([s4, s5, s6, s7, s8])
    is_valid  = hard_ok and (soft_pass >= 3)

    # Weighted confidence score
    w_scores = (
        (2.0 if s1 else 0) +  # saturation — strongest discriminator
        (2.0 if s2 else 0) +  # dark surround — essential for MRI
        (2.0 if s3 else 0) +  # white bg rejection — kills documents
        (1.0 if s4 else 0) +
        (0.8 if s5 else 0) +
        (0.8 if s6 else 0) +
        (0.6 if s7 else 0) +
        (0.8 if s8 else 0)
    )
    confidence = w_scores / (2.0+2.0+2.0+1.0+0.8+0.8+0.6+0.8)
    return is_valid, float(confidence), scores


def mri_gate_ui(is_valid, confidence, reasons, _dk):
    """
    On accepted images: show a clean single-line badge only.
    On rejected images: show the full signal breakdown so the user
    understands exactly why their image was refused.
    """
    pct = int(confidence * 100)

    if is_valid:
        # ── ACCEPTED: minimal clean badge ─────────────────────────
        clr = "#22c55e" if pct >= 80 else "#f59e0b"
        bg  = "rgba(34,197,94,.08)"  if pct >= 80 else "rgba(245,158,11,.07)"
        bdr = "rgba(34,197,94,.35)"  if pct >= 80 else "rgba(245,158,11,.35)"
        st.markdown(f"""
<div style="background:{bg};border:1px solid {bdr};border-radius:10px;
  padding:10px 16px;margin-bottom:12px;
  display:flex;align-items:center;gap:10px;">
  <span style="font-size:18px;">✅</span>
  <div>
    <span style="font-family:'Space Grotesk',sans-serif;font-size:14px;
      font-weight:600;color:{clr};">Brain MRI verified</span>
    <span style="font-family:'DM Mono',monospace;font-size:10px;
      color:{"rgba(255,255,255,.40)" if _dk else "#4a6580"};margin-left:10px;">
      MRI score: {pct}%
    </span>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        # ── REJECTED: full diagnostic breakdown ───────────────────
        rows = ""
        for sig, (passed, detail) in reasons.items():
            if not passed:   # only show FAILED signals to keep it concise
                label = sig.replace("_", " ").title()
                rows += f"""
<div style="display:flex;gap:10px;align-items:flex-start;
  padding:7px 0;border-bottom:1px solid {"rgba(255,255,255,.05)" if _dk else "#f0e8e8"};">
  <span style="color:#ef4444;font-size:13px;flex-shrink:0;">✗</span>
  <span style="font-family:'DM Mono',monospace;font-size:10px;
    color:{"rgba(255,255,255,.70)" if _dk else "#5a2020"};flex:1;line-height:1.6;">
    <strong style="color:#f87171">{label}</strong>&nbsp;—&nbsp;{detail}
  </span>
</div>"""

        st.markdown(f"""
<div style="background:rgba(239,68,68,.08);border:1px solid rgba(239,68,68,.40);
  border-left:4px solid #ef4444;border-radius:12px;
  padding:1.1rem 1.4rem;margin-bottom:1rem;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:.6rem;">
    <span style="font-size:22px;">🚫</span>
    <div>
      <div style="font-family:'Space Grotesk',sans-serif;font-size:15px;
        font-weight:700;color:#f87171;">Image rejected — not a brain MRI</div>
      <div style="font-family:'DM Mono',monospace;font-size:10px;
        color:{"rgba(255,255,255,.45)" if _dk else "#8a3030"};margin-top:2px;">
        MRI confidence score: {pct}% — must be ≥60% with all 3 hard gates passed
      </div>
    </div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:10px;font-weight:600;
    color:{"rgba(255,100,100,.70)" if _dk else "#8a3030"};
    text-transform:uppercase;letter-spacing:.1em;margin-bottom:6px;">
    Failed signals:
  </div>
  <div>{rows}</div>
</div>""", unsafe_allow_html=True)

        st.error(
            "**Input Rejected.** This image does not match the expected "
            "characteristics of an axial brain MRI. Please upload a "
            "T1 or T2-weighted axial brain MRI scan (JPG/PNG/BMP, max 10 MB)."
        )

# ================================================================
# GRAD-CAM - pure CNN, no VLM
# ================================================================
def make_gradcam(model, img_array, pred_idx):
    """
    Compute Grad-CAM via GradientTape:
    1. Find last Conv2D in ResNet50V2 backbone
    2. Record gradients of PRE-SOFTMAX class score w.r.t. feature maps
    3. Pool gradients -> per-channel weights -> weighted sum + ReLU
    4. Normalise to [0,1]
    Returns 7x7 heatmap or None on failure.

    NOTE: Gradients are taken w.r.t. the pre-softmax logit, not the final
    softmax probability. When the model is very confident (softmax output
    near 1.0), gradients of the softmax output itself saturate toward zero,
    which can produce a flat/misleading heatmap for exactly the high-
    confidence predictions clinicians most want explained. Using the
    logit avoids that saturation.
    """
    backbone = next((l for l in model.layers if hasattr(l, "layers")), None)
    last_conv = None
    if backbone:
        last_conv = next((l.name for l in reversed(backbone.layers)
                          if isinstance(l, keras.layers.Conv2D)), None)
    if not last_conv:
        last_conv = next((l.name for l in reversed(model.layers)
                          if isinstance(l, keras.layers.Conv2D)), None)
    if not last_conv:
        return None
    try:
        src = backbone or model

        # Standard trick: temporarily strip the final softmax so gradients
        # are computed w.r.t. the pre-activation logit, then restore it.
        # Falls back to the original (softmax-output) behaviour if the
        # final layer has no swappable `activation` attribute.
        final_layer = model.layers[-1]
        original_activation = getattr(final_layer, "activation", None)
        swapped = False
        if original_activation is not None and \
           getattr(original_activation, "__name__", "") == "softmax":
            final_layer.activation = keras.activations.linear
            swapped = True

        try:
            gm = keras.Model(inputs=model.inputs,
                              outputs=[src.get_layer(last_conv).output, model.output])
            with tf.GradientTape() as tape:
                co, logits = gm(img_array)
                score = logits[:, pred_idx]
            grads = tape.gradient(score, co)
        finally:
            if swapped:
                final_layer.activation = original_activation  # always restore

        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(0, 1, 2))
        hm = tf.nn.relu(tf.reduce_sum(tf.multiply(weights, co[0]), axis=-1)).numpy()
        return (hm / hm.max()) if hm.max() > 0 else hm
    except Exception:
        return None

def synthetic_heatmap(pil_img):
    """
    Demo fallback: derive a spatially meaningful heatmap from MRI intensity.
    Uses brightness + Gaussian bias toward central-right (typical tumor region).
    This is NOT Grad-CAM - it is a demo visualization only.
    """
    g = np.array(pil_img.convert("L").resize((28, 28)), dtype=np.float32)
    g = cv2.GaussianBlur(g, (5, 5), 0)
    m = np.ones_like(g)
    m[:3,:] = m[-3:,:] = m[:,:3] = m[:,-3:] = 0
    h = g * m
    if h.max() > 0: h /= h.max()
    ys, xs = np.mgrid[0:28, 0:28]
    bias = np.exp(-((xs - 16)**2 + (ys - 14)**2) / (2 * 7.5**2))
    h = h * 0.38 + bias * 0.62
    if h.max() > 0: h /= h.max()
    return h

def smooth_hm(raw):
    h = cv2.resize(raw.astype(np.float32), IMG_SIZE)
    h = cv2.GaussianBlur(h, (15, 15), 0)
    return (h / h.max()) if h.max() > 0 else h

def overlay_gradcam(pil_img, hm_raw, alpha=0.55):
    """
    Blend jet colormap over partially desaturated MRI.
    Desaturation (40% colour / 60% gray) mutes the scan so
    red/yellow hotspots dominate exactly as in reference images.
    """
    orig  = np.array(pil_img.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    hm    = smooth_hm(hm_raw)
    hm_c  = (mpl_cm.jet(hm)[:, :, :3] * 255).astype(np.float32)
    gray  = np.mean(orig, axis=2, keepdims=True)
    desat = orig * 0.4 + gray * 0.6
    amask = np.clip(alpha + (1 - alpha) * hm[..., np.newaxis] * 0.5, 0, 1)
    blend = np.clip(desat * (1 - amask) + hm_c * amask, 0, 255).astype(np.uint8)
    return Image.fromarray(blend), hm

# ================================================================
# FIGURE GENERATORS
# ================================================================
def pure_heatmap_fig(hm, pred_class, conf):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(hm, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    ax.axis("off"); ax.set_facecolor("#020609"); fig.patch.set_facecolor("#020609")
    cb = fig.colorbar(im, ax=ax, fraction=0.034, pad=0.02)
    cb.ax.tick_params(colors="#666", labelsize=6)
    cb.set_label("Activation", color="#666", fontsize=7, labelpad=5)
    ax.set_title(f"{pred_class}  {conf:.1f}%", color="#bbb",
                 fontsize=8, pad=7, fontweight="bold")
    plt.tight_layout(pad=0.2)
    return fig

def histogram_fig(hm):
    flat = hm.flatten()
    fig, ax = plt.subplots(figsize=(4, 3.2))
    n, bins, patches = ax.hist(flat, bins=45, edgecolor="none")
    mids = (bins[:-1] + bins[1:]) / 2
    for patch, v in zip(patches, mids):
        patch.set_facecolor(mpl_cm.jet(v)); patch.set_alpha(0.9)
    ax.axvline(flat.mean(),             color="#fbbf24", ls="--", lw=1.3,
               label=f"Mean {flat.mean():.2f}")
    ax.axvline(np.percentile(flat, 90), color="#f87171", ls="--", lw=1.3,
               label=f"P90 {np.percentile(flat,90):.2f}")
    ax.set_xlabel("Activation value", color="#666", fontsize=7)
    ax.set_ylabel("Pixel count",      color="#666", fontsize=7)
    ax.tick_params(colors="#666", labelsize=7)
    ax.set_facecolor("#020609"); fig.patch.set_facecolor("#020609")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2e4a")
    ax.legend(fontsize=6.5, labelcolor="#ccc",
              facecolor="#0a1424", edgecolor="#1a2e4a")
    ax.set_title("Activation Distribution", color="#bbb", fontsize=8, pad=6, fontweight="bold")
    plt.tight_layout(pad=0.4)
    return fig

def prob_fig(preds, pred_idx):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    bars = ax.barh(CLASS_NAMES, preds * 100,
                   color=CLASS_COLORS, height=0.54, edgecolor="none")
    bars[pred_idx].set_edgecolor("#38bdf8"); bars[pred_idx].set_linewidth(2)
    ax.set_xlim(0, 115)
    ax.set_xlabel("Softmax Probability (%)", color="#666", fontsize=8, labelpad=5)
    ax.tick_params(colors="#888", labelsize=8)
    ax.set_facecolor("#020609"); fig.patch.set_facecolor("#020609")
    for sp in ax.spines.values(): sp.set_edgecolor("#1a2e4a")
    for bar, val in zip(bars, preds):
        ax.text(val * 100 + 1.3, bar.get_y() + bar.get_height() / 2,
                f"{val*100:.1f}%", va="center", color="#888", fontsize=8)
    plt.tight_layout(pad=0.5)
    return fig

def four_panel_fig(pil_img, hm_raw, pred_class, conf, demo=False):
    overlay, hm = overlay_gradcam(pil_img, hm_raw)
    orig = np.array(pil_img.convert("RGB").resize(IMG_SIZE))
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("#020609")
    for ax in axes:
        ax.set_facecolor("#020609")
        for sp in ax.spines.values(): sp.set_edgecolor("#1a2e4a")
    axes[0].imshow(orig);               axes[0].axis("off")
    axes[1].imshow(np.array(overlay));  axes[1].axis("off")
    im = axes[2].imshow(hm, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    axes[2].axis("off")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#666", labelsize=6)
    flat = hm.flatten()
    mids = None
    n, bins, patches = axes[3].hist(flat, bins=40, edgecolor="none")
    mids = (bins[:-1] + bins[1:]) / 2
    for p, v in zip(patches, mids):
        p.set_facecolor(mpl_cm.jet(v)); p.set_alpha(0.9)
    axes[3].axvline(flat.mean(), color="#fbbf24", ls="--", lw=1.1)
    axes[3].set_xlabel("Activation", color="#666", fontsize=7)
    axes[3].set_facecolor("#020609")
    for sp in axes[3].spines.values(): sp.set_edgecolor("#1a2e4a")
    axes[3].tick_params(colors="#666", labelsize=6)
    titles = ["Original MRI","Grad-CAM Overlay","Activation Map","Histogram"]
    for ax, t in zip(axes, titles):
        ax.set_title(t, color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    tag = " [DEMO synthetic]" if demo else ""
    fig.suptitle(f"NeuroScan AI  |  Grad-CAM  |  {pred_class} ({conf:.1f}%){tag}",
                 color="#e2e8f0", fontsize=10.5, fontweight="bold", y=1.02)
    plt.tight_layout(pad=0.6)
    return fig

def fig_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0); return buf.read()

def to_b64(img):
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG")
    return base64.standard_b64encode(buf.getvalue()).decode()

# ================================================================
# CLAUDE AI REPORT
# ================================================================
def ai_report(pil_img, pred_class, conf, heatmap_img=None,
              all_probs=None, ensemble_mode="ResNet50V2 single model",
              is_close_call=False, second_cls=None, second_conf=None):
    """
    Calls Claude for a real, image-grounded clinical report.

    Design choices that matter clinically:
    - We do NOT ask Claude to simply "justify" the CNN's top label. Handing
      a vision model the answer up front and asking it to describe
      supporting findings reliably produces confirmatory hallucination —
      it will invent plausible-sounding evidence for whatever label it is
      given, even when the image does not actually support it. Instead,
      Claude is given the FULL probability distribution and is explicitly
      instructed to independently assess whether the image supports the
      top class, a different class in the distribution, or neither.
    - We tell Claude the true, actual model configuration that ran
      (ensemble_mode), not a fixed marketing description, so its report
      doesn't cite a "MobileNetV2 ensemble" that didn't run.
    - We forbid invented precise measurements (exact cm sizes etc.) that
      cannot really be determined by an LLM from a single 2D image without
      a calibrated scale reference — asking for false precision is a
      classic source of confident-but-fabricated report content.
    - On any failure we return is_live=False plus the real error, and the
      caller must show this to the user. Silently substituting the
      hardcoded per-class template without saying so is what made the old
      version look like it was always "just a demo report."
    """
    try:    key = st.secrets["ANTHROPIC_API_KEY"]
    except: key = ""
    if not key:
        return mock_report(pred_class, conf), False, "No ANTHROPIC_API_KEY configured."

    if all_probs is None:
        all_probs = {pred_class: conf / 100.0}
    prob_lines = "\n".join(
        f"  - {name}: {p*100:.1f}%" for name, p in all_probs.items()
    )
    ambiguity_note = (
        f"\nNOTE: This is a CLOSE CALL. The top class ({pred_class}, {conf:.1f}%) "
        f"leads the second class ({second_cls}, {second_conf:.1f}%) by less than "
        f"15 points. Do not present this as a confident single diagnosis — "
        f"your report must explicitly discuss both possibilities."
        if is_close_call else ""
    )

    sysp = """You are a specialist neuro-oncology AI assistant with expertise in brain MRI interpretation, \
acting as a second-reader / decision-support tool for a neurosurgeon — not as the final diagnostic authority.

A CNN has produced a classification result on this axial brain MRI. You are also shown its Grad-CAM \
explainability heatmap (red/yellow = regions the CNN weighted most heavily).

YOUR JOB IS TO INDEPENDENTLY ASSESS THE IMAGE, NOT TO RUBBER-STAMP THE CNN'S TOP LABEL.
- You are given the CNN's full probability distribution across all 4 classes, not just the top pick.
- Look at the actual image and heatmap first. If what you observe best matches the CNN's top class, say so
  and explain why. If what you observe is more consistent with a DIFFERENT class in the distribution (or is
  genuinely ambiguous between two), say that explicitly — do not force-fit a narrative to the top label.
- If the Grad-CAM activation pattern looks anatomically inconsistent with the predicted class (e.g. diffuse
  or off-lesion activation), state that plainly in gradcam_analysis and lower the reliability_score.
- Never invent a precise measurement (e.g. "2.8 cm") that cannot actually be determined from a single 2D
  image without a calibrated scale reference. Use qualitative size descriptors (e.g. "small," "moderate,"
  "large relative to hemisphere") instead, and say measurement requires PACS/calibrated imaging.
- Do not invent enhancement patterns, contrast findings, or sequence type you cannot actually determine from
  a non-contrast/unlabeled image — say "cannot be determined from this image" instead of guessing.

IMPORTANT RULES:
- Respond with valid JSON ONLY. No markdown, no backticks, no preamble.
- Reference specific, real visual features of THIS image and heatmap — not generic textbook descriptions.
- Be honest about limitations and uncertainty. Honesty about not knowing beats false confidence.
- Use proper neuroradiology terminology, but only claim what is actually visible.

JSON schema (fill all fields with substantive, specific, image-grounded content — no placeholder text):
{
  "agrees_with_cnn": true or false — do the visual findings actually support the CNN's top-1 label?,
  "clinical_interpretation": "Detailed description of observed signal characteristics, morphology, mass effect, edema — only what is actually visible. Minimum 3 sentences.",
  "location_morphology": "Anatomical location, relative size (qualitative), shape, borders, compartment (intra/extra-axial) — only if determinable from the image.",
  "model_reasoning": "Assess whether the CNN's top-1 class is visually supported. If you disagree or are uncertain, say so explicitly and explain which class you think better fits, referencing the probability distribution provided.",
  "gradcam_analysis": "Describe which regions show high activation and whether that activation pattern is anatomically appropriate for the predicted class or looks inconsistent/off-target.",
  "risk_level": "HIGH or MODERATE or LOW",
  "risk_justification": "Clinical justification for the risk level given the tumour type under consideration and its typical behavior.",
  "patient_explanation": "Plain English explanation suitable for a patient. Warm, clear, non-alarming but honest. 2-3 sentences. Explicitly note this is AI-assisted, not a diagnosis.",
  "next_steps": "Numbered list of specific recommended clinical actions in priority order.",
  "image_quality": "GOOD or ADEQUATE or POOR",
  "image_quality_notes": "Specific, real observations about this image's quality/resolution/artefacts.",
  "uncertainty_factors": "Specific features of THIS image that reduce prediction confidence or require additional imaging/sequences to resolve.",
  "differential_diagnosis": "1-2 alternative diagnoses genuinely worth considering given what's visible, and why.",
  "reliability_score": 0-100,
  "overall_reliability": "One sentence summary, factoring in whether you agreed with the CNN and how ambiguous the image is.",
  "disclaimer": "AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon."
}"""
    user_text = (
        f"CNN top prediction: {pred_class} ({conf:.1f}% confidence).\n"
        f"Full class probability distribution:\n{prob_lines}\n"
        f"Actual model configuration that produced this result: {ensemble_mode}.\n"
        f"Classes: Glioma, Meningioma, Pituitary Tumor, No Tumor.\n"
        f"{ambiguity_note}\n"
        f"Independently assess the attached MRI and Grad-CAM heatmap and provide your report as JSON only."
    )
    msgs = [
        {"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":to_b64(pil_img)}}
    ]
    if heatmap_img:
        msgs.append({"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":to_b64(heatmap_img)}})
    msgs.append({"type":"text","text":user_text})

    last_err = None
    for attempt in range(2):  # one retry on transient failures (rate limit, brief network blip)
        try:
            client = anthropic.Anthropic(api_key=key)
            r = client.messages.create(
                model="claude-sonnet-5",   # update to whichever current Claude model your account has access to
                max_tokens=1800,
                system=sysp,
                messages=[{"role":"user","content":msgs}],
            )
            raw = r.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.strip("`")
                if raw.lower().startswith("json"):
                    raw = raw[4:]
            parsed = json.loads(raw.strip())
            return parsed, True, None
        except json.JSONDecodeError as e:
            last_err = f"Claude did not return valid JSON: {e}"
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
    # Both attempts failed — return template report BUT tell the caller so the
    # UI can say "template fallback, Claude analysis unavailable" instead of
    # silently presenting canned text as if it were a real read of this image.
    return mock_report(pred_class, conf), False, last_err

def mock_report(pc, c):
    T = {
      "Glioma":{
        "clinical_interpretation":"Heterogeneous mass lesion with irregular margins and peritumoral edema. Mixed signal intensity with areas of necrosis and ring-enhancing pattern characteristic of high-grade glioma. Significant mass effect with midline shift.",
        "location_morphology":"Right frontal lobe, supratentorial compartment. Irregular lobulated borders. Vasogenic edema extends into adjacent white matter tracts.",
        "model_reasoning":f"Glioma ({c:.1f}%) supported by ring-enhancing pattern, heterogeneous signal, and peritumoral edema - hallmarks of high-grade glioblastoma.",
        "gradcam_analysis":"Activation heatmap localised to the tumor epicenter with secondary activation at the peritumoral edema boundary. Model attention is clinically meaningful.",
        "risk_level":"HIGH","risk_justification":"High-grade glioma carries significant morbidity. Urgent multidisciplinary neuro-oncology review is indicated.",
        "patient_explanation":"The scan shows signs of a Glioma brain tumor. This is NOT a final diagnosis - your doctor must confirm with further tests.",
        "next_steps":"1. Neuroradiologist review\n2. Contrast-enhanced MRI\n3. Neurosurgical consultation\n4. Tissue biopsy",
        "image_quality":"GOOD","uncertainty_factors":"Partial ambiguity at tumor-edema boundary.",
        "reliability_score":91,"overall_reliability":"High reliability. Minor uncertainty at infiltrative margin.",
        "image_quality_notes":"T1-weighted axial sequence. Good spatial resolution. No significant motion artefact.",
        "differential_diagnosis":"1. High-grade glioblastoma (GBM). 2. Metastatic lesion — requires contrast enhancement and clinical history.",
        "disclaimer":"AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon.",
      },
      "Meningioma":{
        "clinical_interpretation":"Well-circumscribed extra-axial mass with dural tail sign, homogeneous signal intensity, broad base of attachment along the parasagittal convexity.",
        "location_morphology":"Parasagittal convexity, extra-axial. Broad dural base, smooth well-defined margins. Approximately 2.8 cm.",
        "model_reasoning":f"Meningioma ({c:.1f}%) aligned with extra-axial location, homogeneous signal, and dural attachment.",
        "gradcam_analysis":"Model focuses on the lesion-dura interface and dural tail. Clinically appropriate activation.",
        "risk_level":"MODERATE","risk_justification":"Most meningiomas are WHO Grade I (benign). Risk depends on size and location.",
        "patient_explanation":"The scan suggests a meningioma - usually slow-growing, attached to brain outer lining, often non-cancerous.",
        "next_steps":"1. Neurology review\n2. Contrast-enhanced MRI\n3. Observation vs surgical resection",
        "image_quality":"GOOD","uncertainty_factors":"Cavernous sinus involvement requires dedicated coronal sequences.",
        "reliability_score":86,"overall_reliability":"Good reliability. Dural tail sign increases specificity.",
        "image_quality_notes":"T1-weighted axial sequence. Adequate resolution for lesion characterisation.",
        "differential_diagnosis":"1. Dural metastasis — requires clinical history and contrast MRI. 2. Hemangiopericytoma — less likely given homogeneous signal.",
        "disclaimer":"AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon.",
      },
      "Pituitary Tumor":{
        "clinical_interpretation":"Intrasellar mass expanding the sella turcica with suprasellar extension. Optic chiasm displaced superiorly. Pituitary stalk deviated.",
        "location_morphology":"Sella turcica, approximately 1.6 cm macroadenoma with suprasellar extension. Cavernous sinuses intact.",
        "model_reasoning":f"Pituitary tumor ({c:.1f}%) confirmed by intrasellar location, sella expansion, and chiasm displacement.",
        "gradcam_analysis":"Model activates precisely on the sellar region with secondary activation at the chiasm interface.",
        "risk_level":"MODERATE","risk_justification":"Usually benign pituitary adenoma. Risk from hormonal dysfunction and chiasm compression.",
        "patient_explanation":"The scan shows a tumor in the pituitary gland at the base of the brain. Usually non-cancerous.",
        "next_steps":"1. Endocrinology consultation\n2. Visual field testing\n3. Full hormone panel\n4. Consider surgery",
        "image_quality":"GOOD","uncertainty_factors":"Cavernous sinus invasion requires Knosp grading.",
        "reliability_score":89,"overall_reliability":"High reliability. Sellar location is highly discriminative.",
        "image_quality_notes":"T1-weighted axial sequence. Sellar region adequately visualised.",
        "differential_diagnosis":"1. Craniopharyngioma — typically calcified, more heterogeneous. 2. Rathke cleft cyst — simpler structure, no solid component.",
        "disclaimer":"AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon.",
      },
      "No Tumor":{
        "clinical_interpretation":"Normal brain parenchyma. No mass lesion, abnormal enhancement, or signal abnormality. Age-appropriate cortical and subcortical structures.",
        "location_morphology":"No focal lesion. Gray-white matter differentiation preserved. Midline structures central. Normal ventricles.",
        "model_reasoning":f"No Tumor ({c:.1f}%) consistent with symmetric architecture, no mass effect, preserved sulci and gyri.",
        "gradcam_analysis":"Low distributed activation with no focal pathological concentration - consistent with a normal scan.",
        "risk_level":"LOW","risk_justification":"No imaging evidence of intracranial neoplasm on this study.",
        "patient_explanation":"Good news - the AI did not detect a tumor. The brain scan appears normal. Follow up if symptoms persist.",
        "next_steps":"Clinical follow-up if symptomatic. Repeat imaging if clinically indicated.",
        "image_quality":"GOOD","uncertainty_factors":"None significant.",
        "reliability_score":95,"overall_reliability":"Very high reliability. No focal pathology identified.",
        "image_quality_notes":"T1-weighted axial sequence. Good image quality. Age-appropriate brain parenchyma.",
        "differential_diagnosis":"No significant differential — no mass lesion identified on this sequence.",
        "disclaimer":"AI-assisted decision support only. All findings must be confirmed by a licensed radiologist or neurosurgeon.",
      },
    }
    return T.get(pc, T["Glioma"])

def rb(title, body, v=""):
    return f'<div class="rb {v}"><div class="rb-t">{title}</div><div class="rb-b">{body}</div></div>'

# ================================================================
# SIDEBAR
# ================================================================
with st.sidebar:
    st.markdown("""<p style="font-family:'Space Grotesk',sans-serif;font-size:15px;
      color:#e2e8f0;margin:0 0 1rem;font-weight:600;">Settings</p>""",
                unsafe_allow_html=True)
    try:    _key = st.secrets["ANTHROPIC_API_KEY"]
    except: _key = ""
    use_ai = st.toggle("Claude AI Report", value=bool(_key))
    if _key: st.success("Claude API connected")
    else:    st.error("No API key - template reports")
    if os.path.exists(MODEL_PATH): st.success("CNN Model loaded")
    else:
        st.error(f"Model not found: {MODEL_PATH}")
        st.warning("Demo mode - synthetic heatmaps shown")
    alpha    = st.slider("Heatmap Intensity", 0.2, 0.8, 0.55, 0.05)
    with st.expander("Advanced"):
        temperature = st.slider(
            "Calibration Temperature", 1.0, 2.5, 1.4, 0.1,
            help="Softens/sharpens the probability distribution. T=1.4 was "
                 "picked for the solo ResNet50V2 model — now that the real "
                 "MobileNetV2 ensemble may be active, re-tune this against a "
                 "held-out validation set's reliability diagram rather than "
                 "assuming the old value still applies."
        )
    show_prf = st.toggle("Show Model Performance", value=True)
    st.divider()
    _mob_present    = os.path.exists(MOBILENET_PATH)
    _mob_downloadable = bool(GDRIVE_MOBILENET_ID)
    if _mob_present:
        _ensemble_line = "+MobileNetV2 (loaded)"
    elif _mob_downloadable:
        _ensemble_line = "+MobileNetV2 (will download on first prediction)"
    else:
        _ensemble_line = "None (mobilenet_model.h5 not found, GDRIVE_MOBILENET_FILE_ID not set)"
    st.code(f"Model    : ResNet50V2\nEnsemble : {_ensemble_line}\n"
            "Preproc  : resnet_v2.preprocess_input\nClasses  : 4\n"
            "Input    : 224x224 RGB\nXAI      : Grad-CAM\n"
            "Accuracy : 95.31% (measured WITH the ensemble — see note below)", language="text")
    if not (_mob_present or _mob_downloadable):
        st.caption(
            "⚠️ Running single-model (ResNet50V2 only). The 95.31% accuracy and "
            "the confusion-matrix numbers below were measured WITH the full "
            "ensemble. Solo ResNet50V2 accuracy is lower — expect more Glioma/"
            "Meningioma confusion than the headline number suggests. To run the "
            "real ensemble, add `mobilenet_model.h5` to the app directory or set "
            "the `GDRIVE_MOBILENET_FILE_ID` environment variable."
        )
    st.divider()
    st.markdown("""<div style="background:rgba(245,158,11,.07);border-left:3px solid #f59e0b;
      border-radius:0 8px 8px 0;padding:10px 12px;font-family:'DM Mono',monospace;
      font-size:9.5px;color:rgba(253,211,77,.92);line-height:1.7;">
      <strong style="color:#fbbf24;">Disclaimer</strong><br>
      AI decision support only. Not a substitute for professional medical diagnosis.
      </div>""", unsafe_allow_html=True)

# ================================================================
# TOP NAV + THEME TOGGLE
# ================================================================
_tog_icon      = "☀️" if _dk else "🌙"
_tog_title     = "Switch to light mode" if _dk else "Switch to dark mode"
_next_theme    = "light" if _dk else "dark"
_tog_bg        = "rgba(28,38,68,.85)" if _dk else "#daeeff"
_tog_bdr       = "rgba(56,189,248,.50)" if _dk else "#4a9ab8"
_tog_col       = "#7dd3fc" if _dk else "#084e65"
_tog_hov       = "rgba(56,189,248,.25)" if _dk else "#b8dff0"

# Handle query-param toggle (set by the nav form below)
_qp = st.query_params.get("theme", None)
if _qp in ("light", "dark") and st.session_state.theme != _qp:
    st.session_state.theme = _qp
    st.query_params.clear()
    st.rerun()

st.markdown(f"""
<style>
/* Toggle sits inline in the nav — no Streamlit layout involved */
.theme-toggle {{
  width:34px;height:34px;border-radius:50%;
  background:{_tog_bg};
  border:1px solid {_tog_bdr};
  color:{_tog_col};
  font-size:16px;line-height:1;cursor:pointer;flex-shrink:0;
  display:inline-flex;align-items:center;justify-content:center;
  box-shadow:0 2px 8px rgba(0,0,0,.15);
  transition:background .18s ease,transform .2s ease;
  -webkit-appearance:none;appearance:none;outline:none;
  margin:0;padding:0;
}}
.theme-toggle:hover {{
  background:{_tog_hov};
  transform:scale(1.13) rotate(15deg);
}}
</style>

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
    <span class="chip c-purple">Claude AI Reports</span>
    <form method="get" action="" style="margin:0;padding:0;display:inline-flex;">
      <input type="hidden" name="theme" value="{_next_theme}">
      <button type="submit" class="theme-toggle" title="{_tog_title}" aria-label="{_tog_title}">
        {_tog_icon}
      </button>
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
          Upload any axial brain MRI and receive instant ResNet50V2
          classification across 4 tumor types, complete with Grad-CAM
          heatmaps highlighting the exact regions that drove the prediction,
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
      <div class="pip-step">
        <div class="pip-num">1</div>
        <div class="pip-txt"><strong>Upload MRI</strong>Any axial T1/T2 scan</div>
      </div>
      <div class="pip-arr">›</div>
      <div class="pip-step">
        <div class="pip-num">2</div>
        <div class="pip-txt"><strong>CNN Inference</strong>ResNet50V2 classifies</div>
      </div>
      <div class="pip-arr">›</div>
      <div class="pip-step">
        <div class="pip-num">3</div>
        <div class="pip-txt"><strong>Grad-CAM</strong>Tumor region heatmap</div>
      </div>
      <div class="pip-arr">›</div>
      <div class="pip-step">
        <div class="pip-num">4</div>
        <div class="pip-txt"><strong>AI Report</strong>Clinical analysis</div>
      </div>
      <div class="pip-arr">›</div>
      <div class="pip-step">
        <div class="pip-num">5</div>
        <div class="pip-txt"><strong>Export</strong>JSON + PNG figure</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ================================================================
# CONTENT WRAPPER
# ================================================================
st.markdown('<div class="wrap">', unsafe_allow_html=True)

if not os.path.exists(MODEL_PATH):
    st.info("**Demo Mode** - `brain_tumor_model.h5` not found. "
            "Predictions are simulated and heatmaps are synthetic. "
            "Fix: Streamlit Cloud > Settings > change Branch to `master`.")

# ================================================================
# INPUT / OUTPUT COLUMNS
# ================================================================
col_in, col_out = st.columns([1, 1], gap="large")

with col_in:
    st.markdown('<div class="slbl">Input - MRI Scan</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload MRI",
        type=["jpg","jpeg","png","bmp"],
        label_visibility="collapsed",
        help="Axial T1 or T2-weighted brain MRI. JPEG/PNG up to 10 MB."
    )
    _hint_clr  = "rgba(255,255,255,.68)" if _dk else "#2d4a6b"
    _hint_bg   = "rgba(56,189,248,.06)"  if _dk else "rgba(14,116,144,.05)"
    st.markdown(f'''<div style="font-family:DM Mono,monospace;font-size:10px;
      color:{_hint_clr};text-align:center;padding:8px 0 14px;
      text-transform:uppercase;letter-spacing:.11em;
      background:{_hint_bg};border-radius:8px;margin-top:6px;">
      JPG / PNG / BMP &nbsp;·&nbsp; Max 10 MB &nbsp;·&nbsp; T1 or T2 axial preferred
    </div>''', unsafe_allow_html=True)

    _lbl_clr = "#38bdf8" if _dk else "#0e6b8a"
    _lbl_ln  = "rgba(56,189,248,.40)" if _dk else "#9ecadb"
    st.markdown(f'''<div style="margin:18px 0 8px;">
      <div style="font-family:DM Mono,monospace;font-size:10.5px;font-weight:600;
        color:{_lbl_clr};text-transform:uppercase;letter-spacing:.13em;
        display:flex;align-items:center;gap:10px;">
        <span style="flex:1;height:1px;background:{_lbl_ln};display:block"></span>
        Or choose a pre-loaded sample
        <span style="flex:1;height:1px;background:{_lbl_ln};display:block"></span>
      </div>
    </div>''', unsafe_allow_html=True)

    # Selectbox text fully handled in the main CSS block above
    # This inline override ensures the placeholder specifically is visible
    _sb_bg  = "#1e2d45"               if _dk else "#ffffff"
    _sb_txt = "#f1f5f9"               if _dk else "#0a1628"
    _sb_bdr = "rgba(56,189,248,.55)"  if _dk else "#0e7490"
    _sb_ico = "#38bdf8"               if _dk else "#0e7490"
    st.markdown(f'''<style>
/* Late selectbox override — after all Streamlit styles */
[data-testid="stSelectbox"] > div > div {{
  background:{_sb_bg} !important;
  border:1.5px solid {_sb_bdr} !important;
  border-radius:11px !important;
  min-height:46px !important;
}}
[data-testid="stSelectbox"] > div > div > div,
[data-testid="stSelectbox"] > div > div > div > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div {{
  color:{_sb_txt} !important;
  -webkit-text-fill-color:{_sb_txt} !important;
  font-family:Space Grotesk,sans-serif !important;
  font-size:14px !important;
  font-weight:500 !important;
  background:transparent !important;
}}
[data-testid="stSelectbox"] svg {{ fill:{_sb_ico} !important; }}
</style>''', unsafe_allow_html=True)
    sel_lbl  = st.selectbox("Sample", list(SAMPLES.keys()),
                            index=0, label_visibility="collapsed")
    sel_file = SAMPLES[sel_lbl]

    img = src = None
    if uploaded:
        # Force full pixel decode immediately while buffer is valid.
        # Image.open() is lazy — deferring decode causes corruption
        # when the buffer is consumed by Streamlit's display logic.
        # ImageOps.exif_transpose corrects phone/camera EXIF rotation.
        try:
            from PIL import ImageOps
            import io as _io
            # Read ALL bytes first — guarantees buffer is fully available
            _bytes = uploaded.read()
            _buf   = _io.BytesIO(_bytes)
            _raw   = Image.open(_buf)
            _raw.load()                           # force full pixel decode NOW
            _raw   = ImageOps.exif_transpose(_raw)# correct EXIF rotation
            # Convert to RGB numpy array and back to PIL
            # This is the most reliable way to get a clean, detached image
            import numpy as _np2
            _arr_loaded = _np2.array(_raw.convert("RGB"), dtype=np.uint8)
            img = Image.fromarray(_arr_loaded, mode="RGB")
            src = "upload"
            del _bytes, _buf, _raw, _arr_loaded
        except Exception as _e:
            st.error(f"Failed to open image: {_e}")
            img = None
        if img:
            st.success("Image uploaded successfully.")
    elif sel_file:
        sp = os.path.join(SAMPLE_DIR, sel_file)
        if os.path.exists(sp):
            img = Image.open(sp).convert("RGB"); src = "sample"
        else:
            st.warning(f"Sample not found: `{sp}`")
    else:
        _ep_bdr = "rgba(56,189,248,.38)" if _dk else "#9ecadb"
        _ep_bg  = "rgba(56,189,248,.05)" if _dk else "rgba(14,116,144,.04)"
        _ep_ttl = "#e2e8f0"              if _dk else "#0a1628"
        _ep_sub = "rgba(255,255,255,.58)"if _dk else "#2d4a6b"
        st.markdown(f'''<div style="border:2px dashed {_ep_bdr};border-radius:14px;
          padding:2.5rem 1.5rem;text-align:center;background:{_ep_bg};margin:8px 0;">
          <div style="font-size:40px;margin-bottom:12px;">🩻</div>
          <div style="font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:600;
            color:{_ep_ttl};margin-bottom:6px;">No image selected</div>
          <div style="font-family:DM Mono,monospace;font-size:10px;
            color:{_ep_sub};letter-spacing:.07em;line-height:1.9;">
            Upload a brain MRI above<br>or pick a sample below
          </div>
        </div>''', unsafe_allow_html=True)

    if img:
        cap = "UPLOADED SCAN" if src == "upload" else f"SAMPLE: {sel_lbl.upper()}"
        _cap_clr = "#38bdf8" if _dk else "#0e6b8a"
        st.markdown(f'''<div style="font-family:DM Mono,monospace;font-size:10px;
          font-weight:600;color:{_cap_clr};text-align:center;
          padding:8px 0 4px;letter-spacing:.09em;text-transform:uppercase;">
          📸 {cap}
        </div>''', unsafe_allow_html=True)
        st.image(img, use_column_width=True, clamp=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    _btn_dis_bg  = "rgba(255,255,255,.06)" if _dk else "rgba(0,0,0,.06)"
    _btn_dis_col = "rgba(255,255,255,.30)" if _dk else "rgba(0,0,0,.30)"
    _btn_dis_bdr = "rgba(255,255,255,.08)" if _dk else "rgba(0,0,0,.12)"
    st.markdown(f'''<style>
/* Late-injected button override — wins the specificity race */
div[data-testid="stButton"] > button,
div[data-testid="stButton"] > button[kind] {{
  background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#4f46e5 100%) !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  font-size:16px !important;
  padding:17px 28px !important;
  border-radius:14px !important;
  letter-spacing:.03em !important;
  margin-top:8px !important;
  font-weight:700 !important;
  border:none !important;
  box-shadow:0 6px 24px rgba(37,99,235,.55),0 2px 8px rgba(0,0,0,.20) !important;
  width:100% !important;
  text-shadow:0 1px 2px rgba(0,0,0,.20) !important;
}}
div[data-testid="stButton"] > button:hover {{
  background:linear-gradient(135deg,#38bdf8 0%,#3b82f6 50%,#6366f1 100%) !important;
  box-shadow:0 12px 36px rgba(37,99,235,.65) !important;
  transform:translateY(-2px) !important;
}}
div[data-testid="stButton"] > button:disabled {{
  background:{_btn_dis_bg} !important;
  color:{_btn_dis_col} !important;
  -webkit-text-fill-color:{_btn_dis_col} !important;
  border:1px solid {_btn_dis_bdr} !important;
  cursor:not-allowed !important;
  transform:none !important;
  box-shadow:none !important;
}}
/* Upload button late override */
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button {{
  background:#0f172a !important;
  color:#ffffff !important;
  -webkit-text-fill-color:#ffffff !important;
  border:1.5px solid rgba(255,255,255,.18) !important;
  font-weight:700 !important;
  font-size:14px !important;
  width:auto !important;
  padding:10px 24px !important;
  box-shadow:0 2px 12px rgba(0,0,0,.50) !important;
}}
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover {{
  background:#1e293b !important;
  box-shadow:0 6px 20px rgba(0,0,0,.60) !important;
  transform:translateY(-2px) !important;
}}
</style>''', unsafe_allow_html=True)
    _btn_lbl = "Upload or select an MRI first" if img is None else "Run CNN · Grad-CAM · AI Report"
    clicked = st.button(
        "🔬  Analyze and Generate Clinical Report",
        disabled=(img is None),
        help=_btn_lbl
    )

with col_out:
    st.markdown(f'''<div class="slbl">
      <span id="ns-result-label">
        Model Output - Prediction
      </span>
    </div>''', unsafe_allow_html=True)
    if not clicked:
        _id_ttl  = "#e2e8f0"               if _dk else "#0a1628"
        _id_sub  = "rgba(255,255,255,.65)"  if _dk else "#2d4a6b"
        _id_tbg  = "rgba(56,189,248,.10)"   if _dk else "rgba(14,116,144,.08)"
        _id_tbdr = "rgba(56,189,248,.28)"   if _dk else "#9ecadb"
        _id_tcol = "#7dd3fc"                if _dk else "#0e6b8a"
        _id_tags = "".join([
            f'<span style="font-family:DM Mono,monospace;font-size:9.5px;' +
            f'padding:5px 14px;border-radius:20px;background:{_id_tbg};' +
            f'border:1px solid {_id_tbdr};color:{_id_tcol};letter-spacing:.07em;' +
            f'white-space:nowrap;">{t}</span>'
            for t in ["CNN PREDICTION","GRAD-CAM HEATMAP","CLASS PROBABILITIES","AI CLINICAL REPORT"]
        ])
        st.markdown(f'''<div class="glass" style="min-height:440px;display:flex;
          align-items:center;justify-content:center;text-align:center;padding:3rem;">
          <div>
            <div style="font-size:54px;margin-bottom:18px;">🔬</div>
            <div style="font-family:Space Grotesk,sans-serif;font-size:19px;font-weight:600;
              color:{_id_ttl};line-height:1.4;margin-bottom:8px;">
              Ready for Analysis
            </div>
            <div style="font-family:Inter,sans-serif;font-size:13.5px;
              color:{_id_sub};line-height:1.85;margin-bottom:22px;">
              Upload a brain MRI scan or select a sample,<br>
              then click <strong>Analyse</strong> to run the full pipeline.
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">
              {_id_tags}
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

# ================================================================
# ANALYSIS  (runs unconditionally when clicked)
# ================================================================
if clicked and img:
    model = load_model()

    # ── Auto-scroll to results + sticky result banner ───────────
    # Inject JS that scrolls the Streamlit parent window to the
    # output column, and shows a floating toast notification.
    _result_anchor_id = "ns-result-anchor"
    st.markdown(f"""
<div id="{_result_anchor_id}"></div>
<style>
/* Floating toast notification */
@keyframes ns-slide-in {{
  from {{ transform:translateY(-80px); opacity:0 }}
  to   {{ transform:translateY(0);     opacity:1 }}
}}
@keyframes ns-fade-out {{
  from {{ opacity:1 }} to {{ opacity:0; pointer-events:none }}
}}
#ns-toast {{
  position:fixed; top:70px; left:50%; transform:translateX(-50%);
  z-index:9999;
  background:{"rgba(14,30,70,.97)" if _dk else "rgba(255,255,255,.97)"};
  border:1px solid {"rgba(56,189,248,.50)" if _dk else "#4a9ab8"};
  border-left:4px solid #38bdf8;
  border-radius:12px;
  padding:12px 22px 12px 16px;
  display:flex; align-items:center; gap:12px;
  box-shadow:0 8px 32px rgba(0,0,0,.35);
  animation: ns-slide-in .4s ease forwards, ns-fade-out .5s ease 4s forwards;
  min-width:320px; max-width:480px;
  pointer-events:none;
}}
#ns-toast-icon {{ font-size:22px; flex-shrink:0; }}
#ns-toast-body {{ flex:1 }}
#ns-toast-title {{
  font-family:'Space Grotesk',sans-serif; font-size:14px; font-weight:600;
  color:{"#e2e8f0" if _dk else "#0a1628"}; margin-bottom:2px;
}}
#ns-toast-sub {{
  font-family:'DM Mono',monospace; font-size:10px;
  color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"}; letter-spacing:.05em;
}}
/* Pulsing result indicator on the output column label */
#ns-result-label {{
  display:inline-flex; align-items:center; gap:8px;
}}
#ns-result-dot {{
  width:8px; height:8px; border-radius:50%; background:#38bdf8;
  animation: ns-pulse 1.5s ease infinite;
}}
@keyframes ns-pulse {{
  0%,100% {{ box-shadow:0 0 0 0 rgba(56,189,248,.6); }}
  50%      {{ box-shadow:0 0 0 8px rgba(56,189,248,.0); }}
}}
</style>
<div id="ns-toast">
  <div id="ns-toast-icon">✅</div>
  <div id="ns-toast-body">
    <div id="ns-toast-title">Analysis Complete</div>
    <div id="ns-toast-sub">Results available in the output panel →</div>
  </div>
</div>
<script>
(function() {{
  function scrollToResults() {{
    // Scroll the parent Streamlit page to the output column
    var anchor = document.getElementById('{_result_anchor_id}');
    if (anchor) {{
      anchor.scrollIntoView({{ behavior: 'smooth', block: 'start' }});
    }} else {{
      // Fallback: scroll parent window
      window.parent.scrollBy({{ top: 300, behavior: 'smooth' }});
    }}
  }}
  setTimeout(scrollToResults, 300);
}})();
</script>
""", unsafe_allow_html=True)

    # ── Result ready banner at top of output column ─────────────
    with col_out:
        st.markdown(f"""
<div style="
  background:{"rgba(56,189,248,.08)" if _dk else "rgba(14,116,144,.07)"};
  border:1px solid {"rgba(56,189,248,.30)" if _dk else "#4a9ab8"};
  border-left:4px solid #38bdf8;
  border-radius:12px; padding:12px 16px; margin-bottom:16px;
  display:flex; align-items:center; gap:12px;">
  <span style="font-size:20px">🔬</span>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;
      font-weight:600;color:{"#e2e8f0" if _dk else "#0a1628"};">
      Analysis in progress
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;
      color:{"rgba(255,255,255,.50)" if _dk else "#2d4a6b"};margin-top:2px;
      letter-spacing:.05em;">
      VALIDATION → CNN INFERENCE → GRAD-CAM → AI REPORT
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── MRI Validation Gate ─────────────────────────────────────
    with st.spinner("Validating input image..."):
        _mri_valid, _mri_conf, _mri_reasons = validate_mri(img)

    with col_out:
        mri_gate_ui(_mri_valid, _mri_conf, _mri_reasons, _dk)

    if not _mri_valid:
        st.stop()

    # ── Inference ──────────────────────────────────────────────
    with st.spinner("Running CNN inference..."):
        arr = preprocess(img)  # ResNet50V2 tensor
        if model:
            raw_resnet = model.predict(arr, verbose=0)[0]

            # ── MobileNetV2 ensemble (soft-vote) ─────────────────
            # Only claim an "ensemble" when a genuine second model is
            # actually loaded and averaged. `ensemble_mode` is surfaced
            # to the UI, the sidebar, and the Claude report prompt so
            # none of them overstate what the running system actually did.
            # load_mobilenet() is cached (st.cache_resource) and will pull
            # from GDRIVE_MOBILENET_FILE_ID if the file isn't already on
            # disk — mirrors how the main model is fetched.
            mob_model = load_mobilenet()
            if mob_model is not None:
                try:
                    arr_mob        = preprocess_for_mobilenet(img)  # own preprocessing, not ResNet's
                    raw_mob        = mob_model.predict(arr_mob, verbose=0)[0]
                    raw            = (raw_resnet + raw_mob) / 2.0
                    ensemble_mode  = "ResNet50V2 + MobileNetV2 soft-vote ensemble"
                except Exception:
                    raw           = raw_resnet
                    ensemble_mode = "ResNet50V2 single model (MobileNetV2 inference failed)"
            else:
                # NOTE: Flip/rotation-based "TTA ensemble" has been removed.
                # Horizontal/vertical flips and 180° rotation change the
                # apparent left-right and anterior-posterior position of a
                # lesion. Meningioma vs. glioma discrimination depends on
                # location and extra- vs. intra-axial cues that a CNN not
                # explicitly trained with mirrored augmentation may not
                # handle correctly — mixing flipped-view predictions into
                # the vote can actively pull a correct prediction toward
                # the wrong class. This was a likely contributor to
                # meningioma-as-glioma errors reported on external images.
                #
                # If you want real TTA, only use augmentations that
                # preserve anatomical laterality, e.g. small rotations
                # (±3-5°) or minor brightness/contrast jitter — never
                # flips — and only after confirming empirically (on a
                # held-out validation set) that it improves accuracy.
                raw = raw_resnet
                ensemble_mode = "ResNet50V2 single model (no MobileNetV2 file found, no TTA applied)"

            # ── Temperature scaling ──────────────────────────────
            # T is user-adjustable in the sidebar (Advanced) rather than a
            # hardcoded guess — the right T depends on which model(s) are
            # actually running and should be tuned against a reliability
            # diagram on real validation data, not assumed. This step is
            # mathematically sound: it recovers a pseudo-logit via
            # log(softmax output) and rescales before re-normalising.
            T      = temperature
            logits = np.log(np.clip(raw, 1e-7, 1.0))
            scaled = np.exp(logits / T)
            preds  = scaled / scaled.sum()

            # NOTE: The previous version of this app applied a hardcoded
            # multiplicative "calibration" vector derived from the
            # training-set confusion matrix (e.g. boosting Meningioma by
            # 8% and penalising Glioma by 12% on every single prediction,
            # regardless of image content). That is not valid calibration:
            # it is a static class-prior nudge that does not fix per-image
            # errors, will not generalise to a different external dataset
            # (different class balance = different "correct" bias), and
            # can silently overturn a genuinely confident correct
            # prediction. It has been removed. If Glioma/Meningioma
            # confusion is a known weak spot for this model, surface it
            # as a warning to the clinician (see the ambiguity check
            # below) rather than mathematically overriding the model's
            # own output.
            is_demo = False
        else:
            # Realistic demo predictions with calibrated uncertainty
            _dm = {
                "glioma.jpg":     [0.871, 0.082, 0.027, 0.020],
                "meningioma.jpg": [0.048, 0.891, 0.038, 0.023],
                "no_tumor.jpg":   [0.009, 0.006, 0.978, 0.007],
                "pituitary.jpg":  [0.018, 0.031, 0.014, 0.937],
            }
            # For uploaded images with no model: use a neutral mixed
            # prediction to avoid always showing Glioma.
            # For sample images: use the ground-truth demo dict.
            if src == "upload":
                # Derive pseudo-prediction from image pixel statistics
                # so different uploaded MRIs give different outputs.
                _arr_demo = np.array(img.resize((64,64)).convert("L"), dtype=np.float32)
                _mean_px  = float(_arr_demo.mean())
                _std_px   = float(_arr_demo.std())
                # Heuristic mapping based on intensity profile
                # (approximates what the real model would do)
                if _std_px > 55 and _mean_px < 80:
                    preds = np.array([0.78, 0.12, 0.06, 0.04])  # Glioma-like
                elif _mean_px > 100 and _std_px > 45:
                    preds = np.array([0.06, 0.82, 0.07, 0.05])  # Meningioma-like
                elif _std_px < 35:
                    preds = np.array([0.04, 0.03, 0.91, 0.02])  # No tumor-like
                else:
                    preds = np.array([0.05, 0.07, 0.05, 0.83])  # Pituitary-like
            else:
                _key  = sel_file or "glioma.jpg"
                preds = np.array(_dm.get(_key, _dm["glioma.jpg"]))
            is_demo = True

    # ── Low-confidence warning ──────────────────────────────────
    # Even for accepted images, warn if the CNN is uncertain.
    # This guards against borderline/atypical MRI sequences.

    pidx  = int(np.argmax(preds))
    pcls  = CLASS_NAMES[pidx]
    conf  = float(preds[pidx]) * 100
    rl, rc, dc = RISK[pcls]

    # ── Close-call / ambiguity detection ─────────────────────────
    # Glioma vs. Meningioma is this model's documented weak spot
    # (per its own confusion matrix, ~9.5% of true Gliomas were
    # historically confused with Meningioma). Rather than silently
    # presenting one label with false confidence, flag it when the
    # top-2 classes are close so the clinician sees the ambiguity
    # instead of a single, possibly wrong, answer.
    _sorted_idx   = np.argsort(preds)[::-1]
    _second_idx   = int(_sorted_idx[1])
    _second_cls   = CLASS_NAMES[_second_idx]
    _second_conf  = float(preds[_second_idx]) * 100
    _margin       = conf - _second_conf
    is_close_call = _margin < 15.0  # top-2 within 15 points of each other

    # ── Update toast with actual result ─────────────────────────
    _risk_emoji = {"HIGH": "🔴", "MODERATE": "🟡", "LOW": "🟢"}.get(rl, "🔵")
    st.markdown(f"""
<style>
#ns-toast-icon {{ content:'{_risk_emoji}'; }}
</style>
<script>
(function() {{
  var t = document.getElementById('ns-toast');
  var ti = document.getElementById('ns-toast-title');
  var ts = document.getElementById('ns-toast-sub');
  if (ti) ti.textContent = 'Result: {pcls} ({conf:.1f}%)';
  if (ts) ts.textContent = 'Risk level: {rl} — scroll down for full report';
  // Re-show toast with updated content
  if (t) {{
    t.style.animation = 'none';
    t.offsetHeight; // reflow
    t.style.animation = 'ns-slide-in .4s ease forwards, ns-fade-out .5s ease 5s forwards';
  }}
}})();
</script>
""", unsafe_allow_html=True)

    # ── Low-confidence guard ──────────────────────────────────
    if conf < 55.0:
        with col_out:
            st.warning(
                f"⚠️ **Low Confidence ({conf:.1f}%)** — The ensemble is uncertain. "
                f"Predicted class **{pcls}** but the margin over the next class is "
                f"small. This may indicate an atypical MRI sequence, poor image "
                f"quality, or a genuinely ambiguous case. A specialist review is "
                f"strongly recommended before any clinical decision."
            )
    elif pcls == "Glioma" and conf < 75.0:
        with col_out:
            st.warning(
                f"⚠️ **Borderline Glioma ({conf:.1f}%)** — Glioma and Meningioma "
                f"share overlapping features on non-contrast T1 MRI. This prediction "
                f"confidence is below the reliable threshold for Glioma classification. "
                f"Contrast-enhanced MRI or expert radiological review is advised."
            )
    elif _mri_conf < 0.70:
        with col_out:
            st.info(
                f"ℹ️ MRI gate confidence was {int(_mri_conf*100)}% (borderline). "
                "Some image characteristics were atypical for standard axial MRI. "
                "Verify this is a T1 or T2-weighted axial brain scan."
            )

    # ── External dataset / domain-shift caution ──────────────────
    # This model was trained on one specific source dataset. External MRIs
    # (different scanner, different windowing/contrast, skull-stripped vs.
    # not, different slice plane) live in a different pixel-intensity
    # distribution than what the CNN learned, even when they pass the
    # basic "is this an MRI" gate above. That distribution shift — not a
    # single code bug — is the most common real-world reason a model that
    # scores 95%+ on its own held-out test set misclassifies external
    # images (e.g. meningioma read as glioma). This cannot be fixed by
    # calibration tricks in this app; it requires either (a) retraining /
    # fine-tuning on a sample of the target site's own images, or (b)
    # applying the exact preprocessing pipeline (skull-stripping,
    # intensity normalization) used for the original training data before
    # inference. Surfacing this honestly to the user is better than
    # quietly presenting a possibly-wrong answer with high confidence.
    if src == "upload":
        with col_out:
            st.info(
                "📌 **External image notice** — This model was trained on a specific "
                "MRI dataset. Images from a different scanner, protocol, or "
                "preprocessing pipeline (e.g. not skull-stripped, different "
                "contrast/windowing) can reduce accuracy even when confidence looks "
                "high. If you're validating against a new dataset and seeing "
                "systematic errors (e.g. Meningioma read as Glioma), this is almost "
                "always a data-distribution mismatch, not a bug you can calibrate "
                "away after the fact — the fix is matching preprocessing to training "
                "conditions or fine-tuning on a sample of the new data."
            )

    # ── Grad-CAM - ALWAYS computed (real or synthetic) ─────────
    with st.spinner("Computing Grad-CAM heatmap..."):
        if model:
            raw  = make_gradcam(model, arr, pidx)
            hraw = raw if raw is not None else synthetic_heatmap(img)
            if raw is None: is_demo = True
        else:
            hraw = synthetic_heatmap(img)

        overlay_img, hm = overlay_gradcam(img, hraw, alpha=alpha)

    # ── Stats ───────────────────────────────────────────────────
    mean_a  = float(hm.mean())
    max_a   = float(hm.max())
    p90_a   = float(np.percentile(hm, 90))
    focus_p = float((hm > 0.5).sum() / hm.size * 100)

    # ── AI Report ───────────────────────────────────────────────
    with st.spinner("Generating clinical report..."):
        if use_ai:
            report, report_is_live, report_err = ai_report(
                img, pcls, conf, overlay_img,
                all_probs={n: float(p) for n, p in zip(CLASS_NAMES, preds)},
                ensemble_mode=ensemble_mode if model else "Demo mode (no model loaded)",
                is_close_call=is_close_call,
                second_cls=_second_cls, second_conf=_second_conf,
            )
        else:
            report, report_is_live, report_err = mock_report(pcls, conf), False, None

    # ===========================================================
    # MAIN RESULTS PANEL
    # Exact layout matching reference:
    #   LEFT  col  - prediction name + risk chip + class bars
    #   RIGHT col  - large Grad-CAM overlay (hero image) +
    #                original MRI thumbnail + pure heatmap thumbnail
    # ===========================================================

    # Build per-class bar HTML — all colours theme-aware via _dk
    def class_bar_html(name, prob, is_top, color):
        pct     = prob * 100
        w_pct   = max(pct, 0.5)
        lbl_top  = "font-weight:700;color:#0a1628;"  if not _dk else "font-weight:700;color:#f8fafc;"
        lbl_rest = "font-weight:500;color:#2d4a6b;"  if not _dk else "font-weight:400;color:rgba(255,255,255,.55);"
        bold     = lbl_top if is_top else lbl_rest
        track    = "background:#b8d8ec;" if not _dk else "background:rgba(255,255,255,.07);"
        fill_rst = "background:#7ab8d0;" if not _dk else "background:rgba(255,255,255,.18);"
        fill     = f"background:linear-gradient(90deg,{color},{color}cc);" if is_top else fill_rst
        pct_top  = "color:#0e6b8a;" if not _dk else "color:#38bdf8;"
        pct_rst  = "color:#4a6580;" if not _dk else "color:rgba(255,255,255,.45);"
        pct_col  = pct_top if is_top else pct_rst
        return f"""
<div style="margin-bottom:20px;">
  <div style="font-family:'DM Mono',monospace;font-size:12px;letter-spacing:.12em;
    text-transform:uppercase;margin-bottom:7px;{bold}">{name}</div>
  <div style="{track}border-radius:6px;height:8px;overflow:hidden;margin-bottom:5px;">
    <div style="height:100%;border-radius:6px;width:{w_pct}%;{fill}transition:width .8s ease;"></div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:11px;{pct_col}">{pct:.1f}%</div>
</div>"""


    bars_html = ""
    for i, (cname, cprob, ccol) in enumerate(zip(CLASS_NAMES, preds, CLASS_COLORS)):
        bars_html += class_bar_html(cname, cprob, i == pidx, ccol)

    note = "Synthetic Demo" if is_demo else "ResNet50V2 GradientTape"
    _eyebrow = ("ResNet50V2 + MobileNetV2 Ensemble" if (model and 'ensemble' in ensemble_mode.lower())
                else "ResNet50V2 (single model)")

    with col_out:
        # Prediction name + risk at top of right col
        st.markdown(f"""
<div class="pred-card">
  <div class="pred-eyebrow">{_eyebrow} &nbsp;|&nbsp; 4-Class CNN Prediction</div>
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

    # ===========================================================
    # FULL-WIDTH RESULTS PANEL  (matches reference image exactly)
    # ===========================================================
    st.markdown("---")
    st.markdown("---")
    st.markdown(f"""
<div class="hm-section">
  <div class="hm-header">
    <div>
      <div class="hm-title">Grad-CAM Tumor Region Heatmap &nbsp;|&nbsp; {pcls}</div>
      <div class="hm-sub">{note} &nbsp;|&nbsp; Pure CNN GradientTape &nbsp;|&nbsp;
        No VLM / LLaMA Required &nbsp;|&nbsp;
        Red/yellow = high AI attention / tumor region</div>
    </div>
    <div class="hm-legend">
      <div class="hm-leg">
        <span class="hm-swatch" style="background:linear-gradient(90deg,#00007f,#007fff,#00ffff)"></span>
        Low attention
      </div>
      <div class="hm-leg">
        <span class="hm-swatch" style="background:linear-gradient(90deg,#ffff00,#ff7f00,#ff0000)"></span>
        High attention
      </div>
    </div>
  </div>
""", unsafe_allow_html=True)
    # ── CORE 2-COL LAYOUT: bars left, large heatmap right ──────
    res_left, res_right = st.columns([1, 1], gap="large")

    with res_left:
        # Class name + bar + percentage for every class
        st.markdown(f"""
<div style="padding:1.2rem 0.4rem;">
  <div style="font-family:'DM Mono',monospace;font-size:9px;
    color:{"rgba(56,189,248,.80)" if _dk else "#0e6b8a"};
    text-transform:uppercase;letter-spacing:.16em;margin-bottom:18px;font-weight:600;">
    Class Probability Distribution
  </div>
  {bars_html}
</div>""", unsafe_allow_html=True)

    with res_right:
        # Large Grad-CAM overlay - the hero image
        st.markdown(f"""
<div style="font-family:'DM Mono',monospace;font-size:9.5px;
  color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};
  text-align:right;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">
  Red/yellow = high AI attention
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="hm-img-frame">', unsafe_allow_html=True)
        st.image(overlay_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="text-align:center;margin-top:8px;font-family:'DM Mono',monospace;
  font-size:10px;color:{"rgba(255,255,255,.40)" if _dk else "#4a6580"};letter-spacing:.1em;">
  Grad-CAM Overlay &nbsp;|&nbsp; {pcls}
</div>""", unsafe_allow_html=True)
    # ── SECONDARY ROW: original + pure heatmap + histogram + stats ──
    st.markdown(f'<div style="height:1px;background:{"rgba(255,255,255,.08)" if _dk else "#c2d8e8"};margin:16px 0 14px"></div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1], gap="small")

    with sc1:
        st.markdown('<div class="hm-col-lbl">Original MRI</div>', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown('<div class="hm-col-note">Raw input before preprocessing</div>',
                    unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="hm-col-lbl">Pure Activation Map</div>', unsafe_allow_html=True)
        fh = pure_heatmap_fig(hm, pcls, conf)
        st.pyplot(fh, use_container_width=True); plt.close()
        st.markdown('<div class="hm-col-note">Normalised intensity<br>last conv layer</div>',
                    unsafe_allow_html=True)

    with sc3:
        st.markdown('<div class="hm-col-lbl">Activation Histogram</div>', unsafe_allow_html=True)
        fhist = histogram_fig(hm)
        st.pyplot(fhist, use_container_width=True); plt.close()
        st.markdown('<div class="hm-col-note">Distribution of<br>activation values</div>',
                    unsafe_allow_html=True)

    with sc4:
        st.markdown('<div class="hm-col-lbl">Activation Stats</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:8px;padding-top:2px;">
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{mean_a:.3f}</div>
    <div class="hm-sl">Mean Activation</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{p90_a:.3f}</div>
    <div class="hm-sl">90th Percentile</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{focus_p:.1f}%</div>
    <div class="hm-sl">High-Activation Area</div>
  </div>
  <div class="hm-stat" style="border-radius:8px;border:1px solid rgba(56,189,248,.1)">
    <div class="hm-sv" style="font-size:16px">{max_a:.3f}</div>
    <div class="hm-sl">Peak Activation</div>
  </div>
</div>""", unsafe_allow_html=True)

    # Jet scale + Grad-CAM explainer + close hm-section
    st.markdown(f"""
  <div class="cscale" style="margin:14px 1.5rem;">
    <div class="cscale-bar"></div>
    <div class="cscale-lbls">
      <span>Deep Blue (lowest)</span>
      <span>Cyan</span>
      <span>Green / Yellow</span>
      <span>Orange</span>
      <span>Red (highest / tumor)</span>
    </div>
  </div>

  <div class="hm-explain">
    <div class="hm-exp-title">
      How Grad-CAM Works - Pure CNN Gradient Method, No Language Model Needed
    </div>
    <div class="hm-exp-grid">
      <div class="hm-exp-item">
        <div class="hm-exp-t">Step 1 - Forward Pass</div>
        <div class="hm-exp-b">The 224x224 MRI passes through ResNet50V2.
          The last conv layer outputs a 7x7 feature map with 2048 channels,
          each detecting visual patterns at specific spatial locations.</div>
      </div>
      <div class="hm-exp-item">
        <div class="hm-exp-t">Step 2 - Gradient Backprop</div>
        <div class="hm-exp-b">TensorFlow GradientTape records how strongly each
          feature map channel contributed to the "{pcls}" score.
          Channels that raised the score most get the highest weight.</div>
      </div>
      <div class="hm-exp-item">
        <div class="hm-exp-t">Step 3 - Weighted Sum + ReLU</div>
        <div class="hm-exp-b">Each feature map is multiplied by its weight,
          summed, and ReLU-activated. The coarse 7x7 result is upscaled to
          224x224 via bilinear interpolation.</div>
      </div>
      <div class="hm-exp-item">
        <div class="hm-exp-t">This Scan</div>
        <div class="hm-exp-b">Mean: <strong style="color:#38bdf8">{mean_a:.3f}</strong>,
          Peak: <strong style="color:#38bdf8">{max_a:.3f}</strong>,
          P90: <strong style="color:#38bdf8">{p90_a:.3f}</strong>.
          {focus_p:.1f}% of pixels exceed 0.5 -
          {"focused lesion region." if focus_p < 22 else "broad activation pattern."}</div>
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    if is_demo:
        st.info("**Demo Mode:** Heatmap is synthetic (MRI intensity-derived). "
                "Upload `brain_tumor_model.h5` to master branch for real CNN Grad-CAM gradients.")

    # Download 4-panel
    fig4  = four_panel_fig(img, hraw, pcls, conf, is_demo)
    fbyt  = fig_bytes(fig4); plt.close(fig4)
    st.download_button("Download 4-Panel Grad-CAM Figure (PNG)",
                       data=fbyt,
                       file_name=f"gradcam_{pcls.lower().replace(' ','_')}.png",
                       mime="image/png")

    # ===========================================================
    # CLINICAL REPORT
    # ===========================================================
    st.markdown("---")
    st.markdown('<div class="slbl">AI-Assisted Clinical Report</div>',
                unsafe_allow_html=True)

    if report_is_live:
        st.success("Live Claude analysis of this MRI and its Grad-CAM heatmap "
                    "(not a canned template).")
    else:
        st.warning(
            "Template report shown — this is NOT a live Claude analysis of this "
            "specific image. " + (f"Reason: {report_err}" if report_err else
            "Claude AI Report is toggled off or no API key is configured.")
        )

    if is_close_call:
        st.warning(
            f"Close call: {pcls} ({conf:.1f}%) vs {_second_cls} ({_second_conf:.1f}%) "
            f"— margin under 15 points. Treat this as an ambiguous result requiring "
            f"radiologist review, not a confident single diagnosis."
        )

    if report_is_live and report.get("agrees_with_cnn") is False:
        st.error(
            "Claude's independent read of this image disagrees with the CNN's top "
            "prediction — see Model Reasoning tab below before relying on this result."
        )

    t1, t2, t3, t4 = st.tabs([
        "Clinical Findings", "Model Reasoning", "Patient Summary", "Reliability"
    ])
    with t1:
        st.markdown(rb("Clinical Interpretation",
            report.get("clinical_interpretation",""), "rb-red"), unsafe_allow_html=True)
        st.markdown(rb("Location and Morphology",
            report.get("location_morphology","")), unsafe_allow_html=True)
    with t2:
        st.markdown(rb("Model Reasoning Alignment",
            report.get("model_reasoning","")), unsafe_allow_html=True)
        st.markdown(rb("Grad-CAM Activation Analysis",
            report.get("gradcam_analysis",""), "rb-grn"), unsafe_allow_html=True)
    with t3:
        st.markdown(rb("Plain Language Summary",
            report.get("patient_explanation",""), "rb-yel"), unsafe_allow_html=True)
        st.markdown(rb("Recommended Next Steps",
            report.get("next_steps","").replace("\n","<br>")), unsafe_allow_html=True)
    with t4:
        rs = report.get("reliability_score", 80)
        a, b, c = st.columns(3)
        with a: st.metric("Reliability Score", f"{rs}/100")
        with b: st.metric("Image Quality",     report.get("image_quality","N/A"))
        with c: st.metric("Risk Level",         rl)
        st.progress(rs / 100)
        qv = {"GOOD":"rb-grn","ADEQUATE":"rb-yel","POOR":"rb-red"}.get(
             report.get("image_quality","GOOD"), "rb-grn")
        st.markdown(rb("Uncertainty Factors",
            report.get("uncertainty_factors","None identified."), qv),
            unsafe_allow_html=True)
        st.markdown(rb("Overall Reliability",
            report.get("overall_reliability","")), unsafe_allow_html=True)

    st.markdown(f"""
<div class="disc">
  <strong>AI-Assisted Decision Support Only</strong> -
  {report.get("disclaimer","")}
  This system must not replace professional medical diagnosis.
  All findings require review by a licensed radiologist or neurosurgeon.
</div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)
    gcam_stats = {"mean":round(mean_a,4),"peak":round(max_a,4),
                  "p90":round(p90_a,4),"focus_pct":round(focus_p,2),
                  "synthetic_demo":is_demo}
    st.download_button(
        "Export Full Report (JSON)",
        data=json.dumps({"system":"NeuroScan AI v3.0",
            "model":"ResNet50V2","preprocessing":"resnet_v2.preprocess_input",
            "prediction":pcls,"confidence_pct":round(conf,2),"risk_level":rl,
            "gradcam_stats":gcam_stats,
            "class_probabilities":{n:round(float(p),4) for n,p in zip(CLASS_NAMES,preds)},
            **report}, indent=2),
        file_name=f"neuroscan_{pcls.lower().replace(' ','_')}.json",
        mime="application/json")

    # ===========================================================
    # MODEL PERFORMANCE
    # ===========================================================
    if show_prf:
        st.markdown("---")
        st.markdown('<div class="slbl">Model Performance - Training Results</div>',
                    unsafe_allow_html=True)

        st.markdown("""
<div class="glass" style="margin-bottom:16px;">
  <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(105px,1fr));
    gap:10px;margin-bottom:14px;">
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">95.31%</div><div class="hm-sl">Ensemble Acc.</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">100%</div><div class="hm-sl">No Tumor Recall</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">99.8%</div><div class="hm-sl">Pituitary Recall</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">98.0%</div><div class="hm-sl">Meningioma Recall</div></div>
    <div class="hm-stat" style="border:1px solid rgba(56,189,248,.1);border-radius:10px;padding:10px">
      <div class="hm-sv">83.5%</div><div class="hm-sl">Glioma Recall</div></div>
  </div>
  <div style="font-size:12px;color:rgba(255,255,255,.38);line-height:1.78;">
    Ensemble combines <strong style="color:#38bdf8">ResNet50V2</strong> and
    <strong style="color:#38bdf8">MobileNetV2</strong> via soft-voting of probability
    vectors. ResNet50V2 alone achieves ~93% accuracy; ensemble boosting raises this to
    95.31%. Glioma recall (83.5%) is lower due to visual similarity with meningioma
    on T1 non-contrast sequences. Contrast-enhanced MRI is recommended to confirm findings.
  </div>
</div>
""", unsafe_allow_html=True)

        pt1, pt2 = st.tabs([
            "Training History (ResNet50V2 + MobileNetV2)",
            "Ensemble Confusion Matrix"
        ])
        with pt1:
            if os.path.exists("training_history.png"):
                st.image("training_history.png", use_column_width=True)
            else:
                st.info("`training_history.png` not found - place it in the app root.")
            st.markdown(rb("Reading the Training Curves",
                """<strong>Dashed red line:</strong> Phase 1 to Phase 2 (head layers unfreeze).
                <strong>Dashed orange line:</strong> Phase 2 to Phase 3 (deeper backbone unfreezes).
                The MobileNetV2 dip at Phase 2 is expected - learning rate resets and newly
                unfrozen layers temporarily destabilise before recovering above 97% validation accuracy."""),
                unsafe_allow_html=True)
        with pt2:
            if os.path.exists("confusion_matrix_ensemble.png"):
                st.image("confusion_matrix_ensemble.png", use_column_width=True)
            else:
                st.info("`confusion_matrix_ensemble.png` not found - place it in the app root.")
            st.markdown(rb("Reading the Confusion Matrix",
                """<strong>Left (Counts):</strong> Perfect classification = solid blue diagonal.
                <strong>Right (Recall %):</strong> Per-class recall normalised to 100%.
                No Tumor: <strong>100%</strong>. Pituitary: <strong>99.8%</strong>.
                Meningioma: <strong>98.0%</strong>. Glioma: <strong>83.5%</strong> -
                38 cases confused with Meningioma, a known challenge on T1 non-contrast.""",
                "rb-grn"), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
