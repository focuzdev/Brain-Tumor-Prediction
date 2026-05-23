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
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    resnet_preprocess = None

# ================================================================
# PAGE CONFIG
# ================================================================
st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor MRI Classifier",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)
# Theme is fully client-side (CSS vars + JS localStorage).
# Python only reads it here to render the correct initial icon.
# No reruns are triggered by the toggle.
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ================================================================
# CSS — Pure CSS custom properties, theme toggled entirely client-side
# No Python reruns on theme change. JS flips body.light class + localStorage.
# ================================================================
_dk = (st.session_state.get("theme", "light") == "dark")  # initial icon only

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

:root {
  --bg:          #04090f;
  --bg-ga:       rgba(14,58,150,.45);
  --bg-gb:       rgba(8,100,160,.25);
  --bg-gc:       rgba(5,20,50,.40);
  --txt:         #e2e8f0;
  --txt-m:       rgba(255,255,255,.44);
  --txt-d:       rgba(255,255,255,.24);
  --nav-bg:      rgba(4,9,15,.92);
  --nav-bdr:     rgba(56,189,248,.13);
  --hero-a:      #040c1c;
  --hero-b:      #071630;
  --hero-bdr:    rgba(56,189,248,.09);
  --glass:       rgba(255,255,255,.030);
  --glass-bdr:   rgba(255,255,255,.075);
  --glass-shd:   rgba(0,0,0,.35);
  --slbl:        rgba(56,189,248,.60);
  --slbl-ln:     rgba(56,189,248,.14);
  --rb-bg:       rgba(255,255,255,.028);
  --rb-txt:      rgba(255,255,255,.70);
  --rb-ttl:      rgba(255,255,255,.32);
  --rb-red-bg:   rgba(239,68,68,.04);
  --rb-yel-bg:   rgba(245,158,11,.04);
  --rb-grn-bg:   rgba(16,185,129,.04);
  --disc-bg:     rgba(245,158,11,.05);
  --disc-bdr:    rgba(245,158,11,.20);
  --disc-txt:    rgba(253,211,77,.70);
  --hr:          rgba(255,255,255,.07);
  --tab-bg:      rgba(255,255,255,.040);
  --tab-bdr:     rgba(255,255,255,.065);
  --tab-col:     rgba(255,255,255,.40);
  --met-bg:      rgba(255,255,255,.045);
  --met-bdr:     rgba(255,255,255,.08);
  --met-lbl:     rgba(255,255,255,.40);
  --sel-bg:      rgba(255,255,255,.045);
  --sel-bdr:     rgba(255,255,255,.095);
  --sel-txt:     #e2e8f0;
  --up-bg:       rgba(20,80,130,.22);
  --up-bdr:      rgba(56,189,248,.60);
  --up-txt:      #7dd3fc;
  --code-bg:     rgba(255,255,255,.06);
  --code-txt:    #7dd3fc;
  --code-bdr:    rgba(255,255,255,.07);
  --img-bdr:     rgba(255,255,255,.08);
  --pip-txt:     rgba(255,255,255,.46);
  --pip-str:     rgba(255,255,255,.82);
  --pip-arr:     rgba(56,189,248,.28);
  --hs-lbl:      rgba(255,255,255,.30);
  --hm-bg:       rgba(2,6,14,.97);
  --hm-bdr:      rgba(56,189,248,.16);
  --hm-hdr:      rgba(4,10,24,1);
  --hm-hdr-b:    rgba(56,189,248,.10);
  --hm-div:      rgba(255,255,255,.05);
  --hm-sl:       rgba(255,255,255,.28);
  --hm-lbl:      rgba(255,255,255,.36);
  --hm-note:     rgba(255,255,255,.28);
  --hm-leg:      rgba(255,255,255,.44);
  --hm-img-b:    rgba(255,255,255,.08);
  --hm-ex-bg:    rgba(255,255,255,.018);
  --hm-ex-bdr:   rgba(255,255,255,.06);
  --hm-ex-b:     rgba(255,255,255,.44);
  --cs-bg:       rgba(255,255,255,.022);
  --cs-lbl:      rgba(255,255,255,.32);
  --pr-a:        rgba(14,30,70,.82);
  --pr-b:        rgba(8,20,48,.92);
  --pr-bdr:      rgba(56,189,248,.20);
  --pr-eye:      rgba(56,189,248,.62);
  --pr-name:     #f8fafc;
  --cf-trk:      rgba(255,255,255,.08);
  --cf-l:        rgba(255,255,255,.40);
  --sb-bg:       rgba(4,9,15,.98);
  --sb-bdr:      rgba(56,189,248,.09);
  --tog-bg:      rgba(28,38,68,.80);
  --tog-bdr:     rgba(56,189,248,.42);
  --tog-ico:     #7dd3fc;
  --tog-hov:     rgba(56,189,248,.22);
  --dl-bg:       rgba(56,189,248,.09);
  --dl-bdr:      rgba(56,189,248,.24);
  --dl-txt:      #38bdf8;
  --dl-hov:      rgba(56,189,248,.16);
  --prog-trk:    rgba(255,255,255,.08);
  --succ-bg:     rgba(34,197,94,.08);
  --succ-bdr:    rgba(34,197,94,.25);
  --succ-txt:    #86efac;
  --err-bg:      rgba(239,68,68,.08);
  --err-bdr:     rgba(239,68,68,.25);
  --err-txt:     #fca5a5;
  --warn-bg:     rgba(245,158,11,.08);
  --warn-bdr:    rgba(245,158,11,.25);
  --warn-txt:    #fcd34d;
  --info-bg:     rgba(56,189,248,.07);
  --info-bdr:    rgba(56,189,248,.22);
  --info-txt:    #7dd3fc;
}

/* ══ TRANSITION — smooth on every property that changes ══ */
*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] {
  font-family: 'Inter', sans-serif;
  -webkit-font-smoothing: antialiased;
}
.stApp,
.stApp *,
.topnav,
.hero,
.glass,
.rb, .disc,
[data-testid="stSidebar"],
[data-testid="stFileUploader"],
[data-testid="stMetric"],
.hm-section, .hm-header, .hm-explain, .hm-exp-item,
.pred-card,
.stTabs [data-baseweb="tab-list"] {
  transition:
    background-color .28s ease,
    background .28s ease,
    border-color .22s ease,
    color .22s ease,
    box-shadow .22s ease !important;
}

/* ══ APP BACKGROUND ══ */
.stApp {
  background: var(--bg) !important;
  background-image:
    radial-gradient(ellipse 90% 55% at 10% -5%, var(--bg-ga) 0%, transparent 45%),
    radial-gradient(ellipse 70% 45% at 95% 105%, var(--bg-gb) 0%, transparent 45%),
    radial-gradient(ellipse 50% 70% at 50% 50%, var(--bg-gc) 0%, transparent 70%) !important;
  color: var(--txt) !important;
  min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden }
.block-container { padding: 0 !important; max-width: 100% !important }

/* ══ NAV ══ */
.topnav {
  position: sticky; top: 0; z-index: 200;
  background: var(--nav-bg);
  backdrop-filter: blur(24px) saturate(180%);
  -webkit-backdrop-filter: blur(24px) saturate(180%);
  border-bottom: 1px solid var(--nav-bdr);
  padding: .7rem 2.2rem;
  display: flex; align-items: center; justify-content: space-between; gap: 1rem;
}
.nav-brand { display: flex; align-items: center; gap: 13px }
.nav-logo {
  width: 38px; height: 38px; border-radius: 10px; font-size: 18px;
  background: linear-gradient(135deg,#1e40af,#0e7490);
  display: flex; align-items: center; justify-content: center;
  box-shadow: 0 0 20px rgba(56,189,248,.40); flex-shrink: 0;
}
.nav-name { font-family: 'Space Grotesk',sans-serif; font-size: 16px; font-weight: 700; color: var(--txt); letter-spacing: -.3px }
.nav-name span { color: #38bdf8 }
.nav-tagline { font-family: 'DM Mono',monospace; font-size: 8px; color: var(--txt-d); letter-spacing: .15em; text-transform: uppercase; margin-top: 1px }
.nav-right { display: flex; gap: 6px; flex-wrap: wrap; align-items: center }
.chip { font-family: 'DM Mono',monospace; font-size: 9px; font-weight: 500; padding: 4px 10px; border-radius: 20px; letter-spacing: .06em; text-transform: uppercase; white-space: nowrap }
.c-blue   { background: rgba(59,130,246,.14); color: #93c5fd; border: 1px solid rgba(59,130,246,.28) }
.c-teal   { background: rgba(20,184,166,.14); color: #5eead4; border: 1px solid rgba(20,184,166,.28) }
.c-green  { background: rgba(34,197,94,.14);  color: #86efac; border: 1px solid rgba(34,197,94,.28)  }
.c-amber  { background: rgba(245,158,11,.14); color: #fcd34d; border: 1px solid rgba(245,158,11,.28) }
.c-purple { background: rgba(139,92,246,.14); color: #c4b5fd; border: 1px solid rgba(139,92,246,.28) }

/* ══ THEME TOGGLE BUTTON ══ */
.theme-toggle {
  width: 34px; height: 34px; border-radius: 50%;
  background: var(--tog-bg);
  border: 1px solid var(--tog-bdr);
  color: var(--tog-ico);
  font-size: 16px; line-height: 1;
  display: inline-flex; align-items: center; justify-content: center;
  cursor: pointer; flex-shrink: 0;
  box-shadow: 0 2px 10px rgba(0,0,0,.18);
  transition: background .18s, transform .18s, box-shadow .18s, border-color .18s !important;
  -webkit-appearance: none; appearance: none; outline: none;
  text-decoration: none; user-select: none;
}
.theme-toggle:hover {
  background: var(--tog-hov);
  transform: scale(1.12) rotate(14deg);
  box-shadow: 0 4px 18px rgba(56,189,248,.28);
}

/* ══ HERO ══ */
.hero {
  position: relative; overflow: hidden;
  padding: 3rem 2.4rem 2.6rem;
  background: linear-gradient(130deg, var(--hero-a) 0%, var(--hero-b) 55%, var(--hero-a) 100%);
  border-bottom: 1px solid var(--hero-bdr);
}
.hero::before {
  content: ''; position: absolute; inset: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 60% 80% at 82% 45%, rgba(56,189,248,.07) 0%,transparent 55%),
    radial-gradient(ellipse 40% 60% at 18% 75%, rgba(99,102,241,.06) 0%,transparent 50%),
    repeating-linear-gradient(90deg,transparent,transparent 80px,rgba(255,255,255,.012) 80px,rgba(255,255,255,.012) 81px),
    repeating-linear-gradient(0deg,transparent,transparent 80px,rgba(255,255,255,.012) 80px,rgba(255,255,255,.012) 81px);
}
.hero-inner { position: relative; z-index: 1; max-width: 1400px; margin: 0 auto }
.hero-top { display: flex; align-items: flex-end; justify-content: space-between; flex-wrap: wrap; gap: 1.5rem }
.hero-h1 { font-family: 'Space Grotesk',sans-serif; font-size: 2.55rem; font-weight: 700; color: var(--txt); letter-spacing: -.7px; line-height: 1.13; margin-bottom: .5rem }
.hero-h1 .grad { background: linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text }
.hero-desc { font-size: 13.5px; color: var(--txt-m); line-height: 1.74; max-width: 530px }
.hero-stats { display: flex; gap: 2rem; flex-wrap: wrap; align-items: flex-end }
.hs { text-align: right }
.hs-n { font-family: 'Space Grotesk',sans-serif; font-size: 27px; font-weight: 700; background: linear-gradient(92deg,#38bdf8,#818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; line-height: 1 }
.hs-l { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--hs-lbl); text-transform: uppercase; letter-spacing: .12em; margin-top: 3px }
.hero-div { height: 1px; margin: 1.4rem 0; background: linear-gradient(90deg,rgba(56,189,248,.30),rgba(129,140,248,.18),transparent) }
.pipeline { display: flex; align-items: center; gap: 0; flex-wrap: wrap }
.pip-step { display: flex; align-items: center; gap: 9px }
.pip-num { width: 28px; height: 28px; border-radius: 50%; background: linear-gradient(135deg,#1d4ed8,#0891b2); font-family: 'DM Mono',monospace; font-size: 11px; font-weight: 600; color: #fff; display: flex; align-items: center; justify-content: center; box-shadow: 0 0 14px rgba(56,189,248,.32); flex-shrink: 0 }
.pip-txt { font-size: 11.5px; color: var(--pip-txt); line-height: 1.38 }
.pip-txt strong { color: var(--pip-str); display: block; font-size: 11px }
.pip-arr { color: var(--pip-arr); font-size: 20px; padding: 0 10px }

/* ══ WRAP + GLASS ══ */
.wrap { max-width: 1400px; margin: 0 auto; padding: 2rem 2.2rem 5rem }
.glass { background: var(--glass); border: 1px solid var(--glass-bdr); border-radius: 20px; padding: 1.6rem; backdrop-filter: blur(12px); box-shadow: 0 8px 40px var(--glass-shd) }
.slbl { font-family: 'DM Mono',monospace; font-size: 9.5px; color: var(--slbl); text-transform: uppercase; letter-spacing: .17em; margin-bottom: 12px; display: flex; align-items: center; gap: 8px }
.slbl::after { content: ''; flex: 1; height: 1px; background: var(--slbl-ln) }

/* ══ PREDICTION CARD ══ */
.pred-card { background: linear-gradient(135deg, var(--pr-a), var(--pr-b)); border: 1px solid var(--pr-bdr); border-radius: 18px; padding: 1.5rem 1.6rem; margin-bottom: 14px; position: relative; overflow: hidden; box-shadow: 0 12px 40px rgba(0,0,0,.25) }
.pred-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8,#818cf8) }
.pred-eyebrow { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--pr-eye); text-transform: uppercase; letter-spacing: .18em; margin-bottom: 8px }
.pred-name { font-family: 'Space Grotesk',sans-serif; font-size: 38px; font-weight: 700; color: var(--pr-name); letter-spacing: -1px; line-height: 1.04; margin-bottom: 15px }
.conf-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px }
.conf-l { font-family: 'DM Mono',monospace; font-size: 10px; color: var(--cf-l) }
.conf-v { font-family: 'Space Grotesk',sans-serif; font-size: 14px; color: #38bdf8; font-weight: 600 }
.conf-track { background: var(--cf-trk); border-radius: 8px; height: 6px; overflow: hidden; margin-bottom: 15px }
.conf-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) }
.risk-chip { display: inline-flex; align-items: center; gap: 7px; padding: 6px 15px; border-radius: 20px; font-family: 'DM Mono',monospace; font-size: 10px; font-weight: 500; letter-spacing: .08em; text-transform: uppercase }
.rdot { width: 6px; height: 6px; border-radius: 50% }
.rH  { background: rgba(239,68,68,.13);  color: #fca5a5; border: 1px solid rgba(239,68,68,.30)  }
.rdH { background: #ef4444; box-shadow: 0 0 7px rgba(239,68,68,.5) }
.rM  { background: rgba(245,158,11,.13); color: #fcd34d; border: 1px solid rgba(245,158,11,.30) }
.rdM { background: #f59e0b; box-shadow: 0 0 7px rgba(245,158,11,.5) }
.rL  { background: rgba(34,197,94,.13);  color: #86efac; border: 1px solid rgba(34,197,94,.30)  }
.rdL { background: #22c55e; box-shadow: 0 0 7px rgba(34,197,94,.5) }

/* ══ HEATMAP SECTION ══ */
.hm-section { background: var(--hm-bg); border: 1px solid var(--hm-bdr); border-radius: 22px; overflow: hidden; box-shadow: 0 0 90px rgba(56,189,248,.06), 0 28px 70px rgba(0,0,0,.45); margin: 1.8rem 0 }
.hm-header { background: var(--hm-hdr); border-bottom: 1px solid var(--hm-hdr-b); padding: 1.1rem 1.7rem; display: flex; align-items: flex-start; justify-content: space-between; flex-wrap: wrap; gap: 10px }
.hm-title { font-family: 'Space Grotesk',sans-serif; font-size: 16px; font-weight: 600; color: var(--txt); letter-spacing: -.3px }
.hm-sub { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--hm-sl); text-transform: uppercase; letter-spacing: .13em; margin-top: 3px }
.hm-legend { display: flex; align-items: center; gap: 12px; flex-wrap: wrap }
.hm-leg { display: flex; align-items: center; gap: 6px; font-family: 'DM Mono',monospace; font-size: 9px; color: var(--hm-leg) }
.hm-swatch { width: 28px; height: 9px; border-radius: 3px }
.hm-stats { display: grid; grid-template-columns: repeat(4,1fr); gap: 0; border-bottom: 1px solid var(--hm-div) }
.hm-stat { padding: 13px 16px; border-right: 1px solid var(--hm-div); text-align: center }
.hm-stat:last-child { border-right: none }
.hm-sv { font-family: 'Space Grotesk',sans-serif; font-size: 20px; font-weight: 600; color: #38bdf8; line-height: 1 }
.hm-sl { font-family: 'DM Mono',monospace; font-size: 8.5px; color: var(--hm-sl); text-transform: uppercase; letter-spacing: .1em; margin-top: 4px }
.hm-grid { padding: 1.4rem 1.6rem }
.hm-col-lbl { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--hm-lbl); text-transform: uppercase; letter-spacing: .13em; text-align: center; margin-bottom: 8px }
.hm-col-note { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--hm-note); text-align: center; margin-top: 8px; line-height: 1.6 }
.hm-img-frame { border-radius: 12px; overflow: hidden; border: 1px solid var(--hm-img-b); box-shadow: 0 8px 32px rgba(0,0,0,.40) }
.cscale { background: var(--cs-bg); margin: 0 1.6rem 1.4rem; border-radius: 8px; padding: 9px 13px; border: 1px solid var(--hm-div) }
.cscale-bar { height: 12px; border-radius: 3px; background: linear-gradient(90deg,#00007f 0%,#0000ff 12%,#007fff 24%,#00ffff 36%,#7fff7f 50%,#ffff00 64%,#ff7f00 76%,#ff0000 88%,#7f0000 100%); margin-bottom: 5px }
.cscale-lbls { display: flex; justify-content: space-between; font-family: 'DM Mono',monospace; font-size: 8.5px; color: var(--cs-lbl) }
.hm-explain { background: var(--hm-ex-bg); border-top: 1px solid var(--hm-ex-bdr); padding: 1.3rem 1.7rem }
.hm-exp-title { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--slbl); text-transform: uppercase; letter-spacing: .15em; margin-bottom: 12px }
.hm-exp-grid { display: grid; grid-template-columns: repeat(auto-fit,minmax(190px,1fr)); gap: 10px }
.hm-exp-item { background: var(--hm-ex-bg); border: 1px solid var(--hm-ex-bdr); border-radius: 10px; padding: 12px 14px }
.hm-exp-t { font-family: 'DM Mono',monospace; font-size: 9px; color: #38bdf8; margin-bottom: 5px; font-weight: 500 }
.hm-exp-b { font-size: 11.5px; color: var(--hm-ex-b); line-height: 1.65 }

/* ══ REPORT BLOCKS ══ */
.rb     { border-left: 3px solid rgba(147,197,253,.45); background: var(--rb-bg);     border-radius: 0 12px 12px 0; padding: 14px 18px; margin-bottom: 12px }
.rb-red { border-left-color: rgba(248,113,113,.60);     background: var(--rb-red-bg) }
.rb-yel { border-left-color: rgba(251,191,36,.60);      background: var(--rb-yel-bg) }
.rb-grn { border-left-color: rgba(52,211,153,.60);      background: var(--rb-grn-bg) }
.rb-t { font-family: 'DM Mono',monospace; font-size: 9px; color: var(--rb-ttl); text-transform: uppercase; letter-spacing: .14em; margin-bottom: 7px }
.rb-b { font-size: 13px; line-height: 1.88; color: var(--rb-txt) }

/* ══ DISCLAIMER ══ */
.disc { background: var(--disc-bg); border: 1px solid var(--disc-bdr); border-left: 3px solid rgba(245,158,11,.55); border-radius: 0 12px 12px 0; padding: 13px 18px; font-family: 'DM Mono',monospace; font-size: 10px; color: var(--disc-txt); line-height: 1.78; margin-top: 20px }
.disc strong { color: #fbbf24 }

/* ══ STREAMLIT OVERRIDES ══ */
[data-testid="stSidebar"] { background: var(--sb-bg) !important; border-right: 1px solid var(--sb-bdr) !important }
[data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem !important }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label, [data-testid="stSidebar"] span { color: var(--txt) !important }
.stTabs [data-baseweb="tab-list"] { background: var(--tab-bg) !important; border-radius: 11px !important; padding: 3px !important; gap: 2px !important; border: 1px solid var(--tab-bdr) !important }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-family: 'DM Mono',monospace !important; font-size: 10.5px !important; color: var(--tab-col) !important; padding: 8px 15px !important }
.stTabs [aria-selected="true"] { background: rgba(56,189,248,.15) !important; color: #38bdf8 !important; box-shadow: none !important }
[data-testid="stFileUploader"] { border: 2px dashed var(--up-bdr) !important; border-radius: 16px !important; background: var(--up-bg) !important }
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div { color: var(--up-txt) !important }
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"] { background: linear-gradient(135deg,#1e40af,#0e7490) !important; color: #ffffff !important; border: none !important; border-radius: 10px !important; font-family: 'Space Grotesk',sans-serif !important; font-weight: 600 !important; font-size: 13px !important; padding: 10px 22px !important; box-shadow: 0 4px 18px rgba(14,116,144,.45), 0 0 0 1px rgba(56,189,248,.22) !important; opacity: 1 !important }
[data-testid="stFileUploader"] button:hover,
[data-testid="stFileUploaderDropzone"] button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 28px rgba(14,116,144,.6), 0 0 0 1px rgba(56,189,248,.38), 0 0 30px rgba(56,189,248,.12) !important; filter: brightness(1.1) !important }
.stButton > button { background: linear-gradient(135deg,#1e3a8a 0%,#0e7490 55%,#0891b2 100%) !important; color: #fff !important; border: none !important; border-radius: 13px !important; font-family: 'Space Grotesk',sans-serif !important; font-weight: 600 !important; font-size: 15px !important; padding: 14px 28px !important; width: 100% !important; box-shadow: 0 4px 24px rgba(14,116,144,.4), 0 0 0 1px rgba(56,189,248,.12) !important; transition: all .22s ease !important; letter-spacing: -.1px !important }
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 32px rgba(14,116,144,.55), 0 0 0 1px rgba(56,189,248,.22), 0 0 40px rgba(56,189,248,.1) !important }
.stButton > button:active { transform: translateY(0px) !important }
[data-testid="stSelectbox"] > div > div { background: var(--sel-bg) !important; border: 1px solid var(--sel-bdr) !important; border-radius: 11px !important; color: var(--sel-txt) !important }
[data-testid="stMetric"] { background: var(--met-bg) !important; border: 1px solid var(--met-bdr) !important; border-radius: 14px !important; padding: 13px 16px !important }
[data-testid="stMetricLabel"] { color: var(--met-lbl) !important; font-size: 11px !important }
[data-testid="stMetricValue"] { font-family: 'Space Grotesk',sans-serif !important; color: #38bdf8 !important; font-size: 22px !important }
[data-testid="stProgress"] > div { background: var(--prog-trk) !important; border-radius: 4px !important }
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) !important; border-radius: 4px !important }
[data-testid="stToggle"] label { color: var(--txt-m) !important; font-size: 13px !important }
[data-testid="stSlider"] > div > div > div { background: #1d4ed8 !important }
[data-testid="stDownloadButton"] > button { background: var(--dl-bg) !important; border: 1px solid var(--dl-bdr) !important; color: var(--dl-txt) !important; font-family: 'DM Mono',monospace !important; font-size: 11.5px !important; border-radius: 9px !important; padding: 9px 16px !important; width: 100% !important; box-shadow: none !important }
[data-testid="stDownloadButton"] > button:hover { background: var(--dl-hov) !important; transform: none !important }
code { font-family: 'DM Mono',monospace !important; font-size: 11px !important; background: var(--code-bg) !important; color: var(--code-txt) !important; border: 1px solid var(--code-bdr) !important; border-radius: 4px !important }
hr { border-color: var(--hr) !important; margin: 1.8rem 0 !important }
[data-testid="stImage"] img { border-radius: 12px !important; border: 1px solid var(--img-bdr) !important; display: block !important }
div[data-testid="stSuccess"] { background: var(--succ-bg) !important; border-color: var(--succ-bdr) !important; color: var(--succ-txt) !important }
div[data-testid="stError"]   { background: var(--err-bg)  !important; border-color: var(--err-bdr)  !important; color: var(--err-txt)  !important }
div[data-testid="stWarning"] { background: var(--warn-bg) !important; border-color: var(--warn-bdr) !important; color: var(--warn-txt) !important }
div[data-testid="stInfo"]    { background: var(--info-bg) !important; border-color: var(--info-bdr) !important; color: var(--info-txt) !important }
</style>

<script>
(function() {
  var LIGHT_CSS = [
    ':root {',
    '  --bg:#eef4fb;',
    '  --bg-ga:rgba(186,220,255,.55);',
    '  --bg-gb:rgba(170,220,245,.38);',
    '  --bg-gc:rgba(205,228,250,.30);',
    '  --txt:#0a1628;',
    '  --txt-m:#2d4a6b;',
    '  --txt-d:#4a6580;',
    '  --nav-bg:rgba(255,255,255,.97);',
    '  --nav-bdr:#b0cfe0;',
    '  --hero-a:#daeeff;',
    '  --hero-b:#c4e0f8;',
    '  --hero-bdr:#a8cfe6;',
    '  --glass:rgba(255,255,255,.92);',
    '  --glass-bdr:#b8d4e8;',
    '  --glass-shd:rgba(10,22,80,.08);',
    '  --slbl:#0e6b8a;',
    '  --slbl-ln:#9ecadb;',
    '  --rb-bg:#ffffff;',
    '  --rb-txt:#0a1628;',
    '  --rb-ttl:#2d4a6b;',
    '  --rb-red-bg:#fde8e8;',
    '  --rb-yel-bg:#fef3cd;',
    '  --rb-grn-bg:#d8f5e8;',
    '  --disc-bg:#fef6d8;',
    '  --disc-bdr:#d4a017;',
    '  --disc-txt:#7a4800;',
    '  --hr:#c2d8e8;',
    '  --tab-bg:#ddeef8;',
    '  --tab-bdr:#a8cfe0;',
    '  --tab-col:#1e3a55;',
    '  --met-bg:#ffffff;',
    '  --met-bdr:#a8cfe0;',
    '  --met-lbl:#2d4a6b;',
    '  --sel-bg:#ffffff;',
    '  --sel-bdr:#7ab8d0;',
    '  --sel-txt:#0a1628;',
    '  --up-bg:#d8eefb;',
    '  --up-bdr:#0e7490;',
    '  --up-txt:#084e65;',
    '  --code-bg:#e8f4fb;',
    '  --code-txt:#0a4570;',
    '  --code-bdr:#a0c8dc;',
    '  --img-bdr:#a8cfe0;',
    '  --pip-txt:#2d4a6b;',
    '  --pip-str:#0a1628;',
    '  --pip-arr:#4a8fa8;',
    '  --hs-lbl:#2d4a6b;',
    '  --hm-bg:#f4faff;',
    '  --hm-bdr:#a0cce0;',
    '  --hm-hdr:#daeeff;',
    '  --hm-hdr-b:#a8cfe0;',
    '  --hm-div:#b8d8ec;',
    '  --hm-sl:#2d4a6b;',
    '  --hm-lbl:#1a3550;',
    '  --hm-note:#2d4a6b;',
    '  --hm-leg:#1a3550;',
    '  --hm-img-b:#a8cfe0;',
    '  --hm-ex-bg:#ffffff;',
    '  --hm-ex-bdr:#a8cfe0;',
    '  --hm-ex-b:#0a1628;',
    '  --cs-bg:#ffffff;',
    '  --cs-lbl:#2d4a6b;',
    '  --pr-a:#daeeff;',
    '  --pr-b:#cce5f8;',
    '  --pr-bdr:#7ab8d4;',
    '  --pr-eye:#0e6b8a;',
    '  --pr-name:#0a1628;',
    '  --cf-trk:#b8d8ec;',
    '  --cf-l:#2d4a6b;',
    '  --sb-bg:#f0f8ff;',
    '  --sb-bdr:#a8cfe0;',
    '  --tog-bg:#daeeff;',
    '  --tog-bdr:#4a9ab8;',
    '  --tog-ico:#084e65;',
    '  --tog-hov:#b8dff0;',
    '  --dl-bg:#daeeff;',
    '  --dl-bdr:#4a9ab8;',
    '  --dl-txt:#084e65;',
    '  --dl-hov:#c4d8ec;',
    '  --prog-trk:#b8d8ec;',
    '  --succ-bg:#d4f5e2;',
    '  --succ-bdr:#2e9e5e;',
    '  --succ-txt:#0a4020;',
    '  --err-bg:#fde0e0;',
    '  --err-bdr:#d04040;',
    '  --err-txt:#5e0a0a;',
    '  --warn-bg:#fef3cd;',
    '  --warn-bdr:#c08000;',
    '  --warn-txt:#6a3800;',
    '  --info-bg:#d8eefb;',
    '  --info-bdr:#2878a8;',
    '  --info-txt:#083858;',
    '}'
  ].join('\n');

  var DARK_CSS = [
    ':root {',
    '  --bg:#04090f;',
    '  --bg-ga:rgba(14,58,150,.45);',
    '  --bg-gb:rgba(8,100,160,.25);',
    '  --bg-gc:rgba(5,20,50,.40);',
    '  --txt:#e2e8f0;',
    '  --txt-m:rgba(255,255,255,.44);',
    '  --txt-d:rgba(255,255,255,.24);',
    '  --nav-bg:rgba(4,9,15,.92);',
    '  --nav-bdr:rgba(56,189,248,.13);',
    '  --hero-a:#040c1c;',
    '  --hero-b:#071630;',
    '  --hero-bdr:rgba(56,189,248,.09);',
    '  --glass:rgba(255,255,255,.030);',
    '  --glass-bdr:rgba(255,255,255,.075);',
    '  --glass-shd:rgba(0,0,0,.35);',
    '  --slbl:rgba(56,189,248,.60);',
    '  --slbl-ln:rgba(56,189,248,.14);',
    '  --rb-bg:rgba(255,255,255,.028);',
    '  --rb-txt:rgba(255,255,255,.70);',
    '  --rb-ttl:rgba(255,255,255,.32);',
    '  --rb-red-bg:rgba(239,68,68,.04);',
    '  --rb-yel-bg:rgba(245,158,11,.04);',
    '  --rb-grn-bg:rgba(16,185,129,.04);',
    '  --disc-bg:rgba(245,158,11,.05);',
    '  --disc-bdr:rgba(245,158,11,.20);',
    '  --disc-txt:rgba(253,211,77,.70);',
    '  --hr:rgba(255,255,255,.07);',
    '  --tab-bg:rgba(255,255,255,.040);',
    '  --tab-bdr:rgba(255,255,255,.065);',
    '  --tab-col:rgba(255,255,255,.40);',
    '  --met-bg:rgba(255,255,255,.045);',
    '  --met-bdr:rgba(255,255,255,.08);',
    '  --met-lbl:rgba(255,255,255,.40);',
    '  --sel-bg:rgba(255,255,255,.045);',
    '  --sel-bdr:rgba(255,255,255,.095);',
    '  --sel-txt:#e2e8f0;',
    '  --up-bg:rgba(20,80,130,.22);',
    '  --up-bdr:rgba(56,189,248,.60);',
    '  --up-txt:#7dd3fc;',
    '  --code-bg:rgba(255,255,255,.06);',
    '  --code-txt:#7dd3fc;',
    '  --code-bdr:rgba(255,255,255,.07);',
    '  --img-bdr:rgba(255,255,255,.08);',
    '  --pip-txt:rgba(255,255,255,.46);',
    '  --pip-str:rgba(255,255,255,.82);',
    '  --pip-arr:rgba(56,189,248,.28);',
    '  --hs-lbl:rgba(255,255,255,.30);',
    '  --hm-bg:rgba(2,6,14,.97);',
    '  --hm-bdr:rgba(56,189,248,.16);',
    '  --hm-hdr:rgba(4,10,24,1);',
    '  --hm-hdr-b:rgba(56,189,248,.10);',
    '  --hm-div:rgba(255,255,255,.05);',
    '  --hm-sl:rgba(255,255,255,.28);',
    '  --hm-lbl:rgba(255,255,255,.36);',
    '  --hm-note:rgba(255,255,255,.28);',
    '  --hm-leg:rgba(255,255,255,.44);',
    '  --hm-img-b:rgba(255,255,255,.08);',
    '  --hm-ex-bg:rgba(255,255,255,.018);',
    '  --hm-ex-bdr:rgba(255,255,255,.06);',
    '  --hm-ex-b:rgba(255,255,255,.44);',
    '  --cs-bg:rgba(255,255,255,.022);',
    '  --cs-lbl:rgba(255,255,255,.32);',
    '  --pr-a:rgba(14,30,70,.82);',
    '  --pr-b:rgba(8,20,48,.92);',
    '  --pr-bdr:rgba(56,189,248,.20);',
    '  --pr-eye:rgba(56,189,248,.62);',
    '  --pr-name:#f8fafc;',
    '  --cf-trk:rgba(255,255,255,.08);',
    '  --cf-l:rgba(255,255,255,.40);',
    '  --sb-bg:rgba(4,9,15,.98);',
    '  --sb-bdr:rgba(56,189,248,.09);',
    '  --tog-bg:rgba(28,38,68,.80);',
    '  --tog-bdr:rgba(56,189,248,.42);',
    '  --tog-ico:#7dd3fc;',
    '  --tog-hov:rgba(56,189,248,.22);',
    '  --dl-bg:rgba(56,189,248,.09);',
    '  --dl-bdr:rgba(56,189,248,.24);',
    '  --dl-txt:#38bdf8;',
    '  --dl-hov:rgba(56,189,248,.16);',
    '  --prog-trk:rgba(255,255,255,.08);',
    '  --succ-bg:rgba(34,197,94,.08);',
    '  --succ-bdr:rgba(34,197,94,.25);',
    '  --succ-txt:#86efac;',
    '  --err-bg:rgba(239,68,68,.08);',
    '  --err-bdr:rgba(239,68,68,.25);',
    '  --err-txt:#fca5a5;',
    '  --warn-bg:rgba(245,158,11,.08);',
    '  --warn-bdr:rgba(245,158,11,.25);',
    '  --warn-txt:#fcd34d;',
    '  --info-bg:rgba(56,189,248,.07);',
    '  --info-bdr:rgba(56,189,248,.22);',
    '  --info-txt:#7dd3fc;',
    '}'
  ].join('\n');

  // ── Inject a <style> into the PARENT document (the real Streamlit page) ──
  var pd = window.parent.document;

  function injectParentStyle(css) {
    var existing = pd.getElementById('neuroscan-theme-vars');
    if (existing) { existing.textContent = css; return; }
    var tag = pd.createElement('style');
    tag.id = 'neuroscan-theme-vars';
    tag.textContent = css;
    pd.head.appendChild(tag);
  }

  // ── Apply theme to parent body ────────────────────────────────
  function applyTheme(isDark) {
    injectParentStyle(isDark ? DARK_CSS : LIGHT_CSS);
    pd.body.classList.toggle('neuroscan-dark',  isDark);
    pd.body.classList.toggle('neuroscan-light', !isDark);
  }

  // ── Restore saved preference; default = LIGHT ─────────────────
  var saved  = localStorage.getItem('neuroscan-theme');
  var isDark = (saved === 'dark');   // default light unless explicitly saved as dark
  applyTheme(isDark);

  // ── Wire the toggle button (retries until nav renders) ─────────
  function wireToggle() {
    // Button lives in an iframe, but we can reach it from here
    var btn = document.getElementById('theme-toggle-btn');
    if (!btn) { setTimeout(wireToggle, 80); return; }

    // Set correct initial icon to match actual applied theme
    btn.textContent = isDark ? '☀️' : '🌙';
    btn.title       = isDark ? 'Switch to light mode' : 'Switch to dark mode';

    btn.addEventListener('click', function() {
      isDark = !isDark;
      applyTheme(isDark);
      localStorage.setItem('neuroscan-theme', isDark ? 'dark' : 'light');
      btn.textContent = isDark ? '☀️' : '🌙';
      btn.title       = isDark ? 'Switch to light mode' : 'Switch to dark mode';
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', wireToggle);
  } else {
    wireToggle();
  }
})();
</script>
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

# ================================================================
# PREPROCESSING - ResNet50V2 expects ImageNet channel stats, NOT /255
# ================================================================
def preprocess(img):
    arr = np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32)
    if resnet_preprocess is not None:
        arr = resnet_preprocess(arr)
    else:
        arr = arr / 127.5 - 1.0
    return np.expand_dims(arr, 0)

# ================================================================
# GRAD-CAM - pure CNN, no VLM
# ================================================================
def make_gradcam(model, img_array, pred_idx):
    """
    Compute Grad-CAM via GradientTape:
    1. Find last Conv2D in ResNet50V2 backbone
    2. Record gradients of class score w.r.t. feature maps
    3. Pool gradients -> per-channel weights -> weighted sum + ReLU
    4. Normalise to [0,1]
    Returns 7x7 heatmap or None on failure.
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
        gm  = keras.Model(inputs=model.inputs,
                          outputs=[src.get_layer(last_conv).output, model.output])
        with tf.GradientTape() as tape:
            co, logits = gm(img_array)
            score = logits[:, pred_idx]
        grads   = tape.gradient(score, co)
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
def ai_report(pil_img, pred_class, conf, heatmap_img=None):
    try:    key = st.secrets["ANTHROPIC_API_KEY"]
    except: key = ""
    if not key: return mock_report(pred_class, conf)
    try:
        client = anthropic.Anthropic(api_key=key)
        sysp   = """Expert neuro-oncology AI. Valid JSON only (no markdown):
{"clinical_interpretation":"...","location_morphology":"...","model_reasoning":"...",
"gradcam_analysis":"...","risk_level":"HIGH|MODERATE|LOW","risk_justification":"...",
"patient_explanation":"...","next_steps":"...","image_quality":"GOOD|ADEQUATE|POOR",
"uncertainty_factors":"...","reliability_score":0-100,"overall_reliability":"...",
"disclaimer":"AI-assisted decision support only."}"""
        msgs = [{"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":to_b64(pil_img)}}]
        if heatmap_img:
            msgs.append({"type":"image","source":{"type":"base64","media_type":"image/jpeg","data":to_b64(heatmap_img)}})
        msgs.append({"type":"text","text":
            f"CNN: {pred_class} ({conf:.1f}%). Classes: Glioma, Meningioma, Pituitary Tumor, No Tumor. JSON only."})
        r = client.messages.create(model="claude-sonnet-4-5", max_tokens=1500,
                                   system=sysp, messages=[{"role":"user","content":msgs}])
        raw = r.content[0].text.strip().replace("```json","").replace("```","").strip()
        return json.loads(raw)
    except:
        return mock_report(pred_class, conf)

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
        "disclaimer":"AI-assisted decision support only.",
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
        "reliability_score":86,"overall_reliability":"Good reliability.",
        "disclaimer":"AI-assisted decision support only.",
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
        "reliability_score":89,"overall_reliability":"High reliability.",
        "disclaimer":"AI-assisted decision support only.",
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
        "reliability_score":95,"overall_reliability":"Very high reliability.",
        "disclaimer":"AI-assisted decision support only.",
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
    show_prf = st.toggle("Show Model Performance", value=True)
    st.divider()
    st.code("Model    : ResNet50V2\nEnsemble : +MobileNetV2\n"
            "Preproc  : resnet_v2.preprocess_input\nClasses  : 4\n"
            "Input    : 224x224 RGB\nXAI      : Grad-CAM\n"
            "Accuracy : 95.31%", language="text")
    st.divider()
    st.markdown("""<div style="background:rgba(245,158,11,.07);border-left:3px solid #f59e0b;
      border-radius:0 8px 8px 0;padding:10px 12px;font-family:'DM Mono',monospace;
      font-size:9.5px;color:rgba(253,211,77,.65);line-height:1.7;">
      <strong style="color:#fbbf24;">Disclaimer</strong><br>
      AI decision support only. Not a substitute for professional medical diagnosis.
      </div>""", unsafe_allow_html=True)

# ================================================================
# TOP NAV + THEME TOGGLE
# ================================================================
# _dk gives the server-side initial icon. JS swaps it client-side after that.
_init_icon  = "🌙" if not _dk else "☀️"
_init_title = "Switch to dark mode" if not _dk else "Switch to light mode"

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
    <span class="chip c-purple">Claude AI Reports</span>
    <button id="theme-toggle-btn"
            class="theme-toggle"
            title="{_init_title}"
            aria-label="{_init_title}">{_init_icon}</button>
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
        <h1 class="hero-h1">AI-Powered Brain Tumor<br>
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
    st.markdown("""<p style="font-family:'DM Mono',monospace;font-size:9px;
      color:rgba(255,255,255,.22);text-align:center;padding:4px 0 10px;
      text-transform:uppercase;letter-spacing:.12em;">
      JPG / PNG / BMP &nbsp;·&nbsp; Max 10 MB &nbsp;·&nbsp; T1 or T2 axial preferred
    </p>""", unsafe_allow_html=True)

    st.markdown("""<p style="font-family:'DM Mono',monospace;font-size:9px;
      color:rgba(56,189,248,.45);text-transform:uppercase;letter-spacing:.14em;
      margin-bottom:6px;">Or choose a pre-loaded sample</p>""",
                unsafe_allow_html=True)

    sel_lbl  = st.selectbox("Sample", list(SAMPLES.keys()),
                            index=0, label_visibility="collapsed")
    sel_file = SAMPLES[sel_lbl]

    img = src = None
    if uploaded:
        img = Image.open(uploaded); src = "upload"
        st.success("Image uploaded successfully.")
    elif sel_file:
        sp = os.path.join(SAMPLE_DIR, sel_file)
        if os.path.exists(sp):
            img = Image.open(sp); src = "sample"
        else:
            st.warning(f"Sample not found: `{sp}`")
    else:
        st.markdown("""
<div style="border:1.5px dashed rgba(56,189,248,.14);border-radius:12px;
  padding:2.5rem;text-align:center;background:rgba(56,189,248,.02);margin:8px 0">
  <div style="font-size:34px;margin-bottom:10px;opacity:.3">🩻</div>
  <div style="font-family:'DM Mono',monospace;font-size:9.5px;
    color:rgba(255,255,255,.22);letter-spacing:.09em;line-height:1.8;">
    Upload an MRI image above<br>or select a pre-loaded sample
  </div>
</div>""", unsafe_allow_html=True)

    if img:
        cap = "UPLOADED SCAN" if src == "upload" else f"SAMPLE: {sel_lbl.upper()}"
        st.image(img, caption=cap, use_column_width=True, clamp=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    clicked = st.button("Analyze and Generate Clinical Report",
                        disabled=(img is None))

with col_out:
    st.markdown('<div class="slbl">Model Output - Prediction</div>', unsafe_allow_html=True)
    if not clicked:
        st.markdown("""
<div class="glass" style="min-height:400px;display:flex;
  align-items:center;justify-content:center;text-align:center;">
  <div>
    <div style="font-size:50px;margin-bottom:16px;opacity:.18">🔬</div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:17px;
      color:rgba(255,255,255,.2);font-weight:300;line-height:1.7;">
      Upload an MRI or select a sample,<br>then click Analyze.
    </div>
    <div style="font-family:'DM Mono',monospace;font-size:9px;
      color:rgba(255,255,255,.12);margin-top:12px;line-height:2;">
      CNN PREDICTION &nbsp;|&nbsp; GRAD-CAM HEATMAP<br>
      CLASS PROBABILITIES &nbsp;|&nbsp; AI CLINICAL REPORT
    </div>
  </div>
</div>""", unsafe_allow_html=True)

# ================================================================
# ANALYSIS  (runs unconditionally when clicked)
# ================================================================
if clicked and img:
    model = load_model()

    # ── Inference ──────────────────────────────────────────────
    with st.spinner("Running CNN inference..."):
        arr = preprocess(img)
        if model:
            preds   = model.predict(arr, verbose=0)[0]
            is_demo = False
        else:
            _dm = {
                "glioma.jpg":     [0.942, 0.031, 0.019, 0.008],
                "meningioma.jpg": [0.052, 0.876, 0.048, 0.024],
                "no_tumor.jpg":   [0.012, 0.009, 0.968, 0.011],
                "pituitary.jpg":  [0.021, 0.043, 0.021, 0.915],
            }
            preds   = np.array(_dm.get(sel_file or "glioma.jpg", _dm["glioma.jpg"]))
            is_demo = True

    pidx  = int(np.argmax(preds))
    pcls  = CLASS_NAMES[pidx]
    conf  = float(preds[pidx]) * 100
    rl, rc, dc = RISK[pcls]

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
        report = (ai_report(img, pcls, conf, overlay_img)
                  if use_ai else mock_report(pcls, conf))

    # ===========================================================
    # MAIN RESULTS PANEL
    # Exact layout matching reference:
    #   LEFT  col  - prediction name + risk chip + class bars
    #   RIGHT col  - large Grad-CAM overlay (hero image) +
    #                original MRI thumbnail + pure heatmap thumbnail
    # ===========================================================

    # Build per-class bar HTML for every class
    def class_bar_html(name, prob, is_top, color):
        pct   = prob * 100
        w_pct = max(pct, 0.5)
        bold  = "font-weight:700;color:#f8fafc;" if is_top else "font-weight:400;color:rgba(255,255,255,.55);"
        return f"""
<div style="margin-bottom:20px;">
  <div style="font-family:'DM Mono',monospace;font-size:12px;letter-spacing:.12em;
    text-transform:uppercase;margin-bottom:7px;{bold}">{name}</div>
  <div style="background:rgba(255,255,255,.07);border-radius:6px;
    height:8px;overflow:hidden;margin-bottom:5px;">
    <div style="height:100%;border-radius:6px;width:{w_pct}%;
      background:{'linear-gradient(90deg,'+color+','+color+'cc)' if is_top else 'rgba(255,255,255,.18)'};
      transition:width .8s ease;"></div>
  </div>
  <div style="font-family:'DM Mono',monospace;font-size:11px;
    color:{'#38bdf8' if is_top else 'rgba(255,255,255,.35)'};">{pct:.1f}%</div>
</div>"""

    bars_html = ""
    for i, (cname, cprob, ccol) in enumerate(zip(CLASS_NAMES, preds, CLASS_COLORS)):
        bars_html += class_bar_html(cname, cprob, i == pidx, ccol)

    note = "Synthetic Demo" if is_demo else "CNN GradientTape"

    with col_out:
        # Prediction name + risk at top of right col
        st.markdown(f"""
<div class="pred-card">
  <div class="pred-eyebrow">ResNet50V2 &nbsp;|&nbsp; 4-Class CNN Prediction</div>
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
  <div style="font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.5);
    text-transform:uppercase;letter-spacing:.16em;margin-bottom:18px;">
    Class Probability Distribution
  </div>
  {bars_html}
</div>""", unsafe_allow_html=True)

    with res_right:
        # Large Grad-CAM overlay - the hero image
        st.markdown("""
<div style="font-family:'DM Mono',monospace;font-size:9.5px;color:rgba(255,255,255,.35);
  text-align:right;letter-spacing:.1em;text-transform:uppercase;margin-bottom:6px;">
  Red/yellow regions = high AI attention
</div>""", unsafe_allow_html=True)
        st.markdown('<div class="hm-img-frame">', unsafe_allow_html=True)
        st.image(overlay_img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown(f"""
<div style="text-align:center;margin-top:8px;font-family:'DM Mono',monospace;
  font-size:10px;color:rgba(255,255,255,.3);letter-spacing:.1em;">
  Grad-CAM Overlay &nbsp;|&nbsp; {pcls}
</div>""", unsafe_allow_html=True)

    # ── SECONDARY ROW: original + pure heatmap + histogram + stats ──
    st.markdown('<div style="height:1px;background:rgba(255,255,255,.05);margin:16px 0 14px"></div>',
                unsafe_allow_html=True)

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
