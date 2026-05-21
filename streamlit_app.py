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
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

st.set_page_config(
    page_title="NeuroScan AI | Brain Tumor MRI Classifier",
    page_icon="\U0001f9e0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Resolve theme from query param BEFORE CSS renders ───────────
# The nav toggle is a plain <a href="?theme=light"> — no widget needed.
# On click Streamlit reruns; we read the param here, update session_state,
# clear the URL, and rerun once more so CSS gets the correct palette.
_qp_theme = st.query_params.get("theme", None)
if _qp_theme in ("light", "dark") and st.session_state.get("theme", "dark") != _qp_theme:
    st.session_state.theme = _qp_theme
    st.query_params.clear()
    st.rerun()

# ================================================================
# CSS — Theme-aware clinical design (all values injected server-side)
# ================================================================
_T  = st.session_state.get("theme", "dark")
_dk = (_T == "dark")

_bg          = "#04090f"                       if _dk else "#f0f6ff"
_bg_ga       = "rgba(14,58,150,.45)"           if _dk else "rgba(186,220,255,.60)"
_bg_gb       = "rgba(8,100,160,.25)"           if _dk else "rgba(170,220,245,.40)"
_bg_gc       = "rgba(5,20,50,.40)"             if _dk else "rgba(205,228,250,.35)"
_txt         = "#e2e8f0"                       if _dk else "#0d1b2e"
_txt_m       = "rgba(255,255,255,.44)"         if _dk else "rgba(13,27,46,.58)"
_txt_d       = "rgba(255,255,255,.24)"         if _dk else "rgba(13,27,46,.38)"
_nav_bg      = "rgba(4,9,15,.92)"              if _dk else "rgba(245,251,255,.96)"
_nav_bdr     = "rgba(56,189,248,.13)"          if _dk else "rgba(14,116,144,.16)"
_hero_a      = "#040c1c"                       if _dk else "#e4f2ff"
_hero_b      = "#071630"                       if _dk else "#cfe7fb"
_hero_bdr    = "rgba(56,189,248,.09)"          if _dk else "rgba(14,116,144,.14)"
_glass       = "rgba(255,255,255,.030)"        if _dk else "rgba(255,255,255,.80)"
_glass_bdr   = "rgba(255,255,255,.075)"        if _dk else "rgba(14,116,144,.16)"
_glass_shd   = "rgba(0,0,0,.35)"              if _dk else "rgba(14,116,144,.10)"
_slbl        = "rgba(56,189,248,.60)"          if _dk else "rgba(14,116,144,.75)"
_slbl_ln     = "rgba(56,189,248,.14)"          if _dk else "rgba(14,116,144,.20)"
_rb_bg       = "rgba(255,255,255,.028)"        if _dk else "rgba(255,255,255,.72)"
_rb_txt      = "rgba(255,255,255,.70)"         if _dk else "rgba(13,27,46,.72)"
_rb_ttl      = "rgba(255,255,255,.32)"         if _dk else "rgba(13,27,46,.40)"
_disc_bg     = "rgba(245,158,11,.05)"          if _dk else "rgba(254,243,199,.72)"
_disc_bdr    = "rgba(245,158,11,.20)"          if _dk else "rgba(180,120,0,.25)"
_disc_txt    = "rgba(253,211,77,.70)"          if _dk else "rgba(100,60,0,.80)"
_hr          = "rgba(255,255,255,.07)"         if _dk else "rgba(14,116,144,.13)"
_tab_bg      = "rgba(255,255,255,.040)"        if _dk else "rgba(224,241,252,.75)"
_tab_bdr     = "rgba(255,255,255,.065)"        if _dk else "rgba(14,116,144,.15)"
_tab_col     = "rgba(255,255,255,.40)"         if _dk else "rgba(13,27,46,.48)"
_met_bg      = "rgba(255,255,255,.045)"        if _dk else "rgba(255,255,255,.82)"
_met_bdr     = "rgba(255,255,255,.08)"         if _dk else "rgba(14,116,144,.14)"
_met_lbl     = "rgba(255,255,255,.40)"         if _dk else "rgba(13,27,46,.44)"
_sel_bg      = "rgba(255,255,255,.045)"        if _dk else "rgba(255,255,255,.82)"
_sel_bdr     = "rgba(255,255,255,.095)"        if _dk else "rgba(14,116,144,.18)"
_sel_txt     = "#e2e8f0"                       if _dk else "#0d1b2e"
_up_bg       = "rgba(20,80,130,.22)"           if _dk else "rgba(214,240,254,.90)"
_up_bdr      = "rgba(56,189,248,.60)"          if _dk else "rgba(14,116,144,.65)"
_up_txt      = "#7dd3fc"                       if _dk else "#0e7490"
_code_bg     = "rgba(255,255,255,.06)"         if _dk else "rgba(14,116,144,.07)"
_code_txt    = "#7dd3fc"                       if _dk else "#0369a1"
_code_bdr    = "rgba(255,255,255,.07)"         if _dk else "rgba(14,116,144,.18)"
_img_bdr     = "rgba(255,255,255,.08)"         if _dk else "rgba(14,116,144,.16)"
_pip_txt     = "rgba(255,255,255,.46)"         if _dk else "rgba(13,27,46,.52)"
_pip_str     = "rgba(255,255,255,.82)"         if _dk else "rgba(13,27,46,.86)"
_pip_arr     = "rgba(56,189,248,.28)"          if _dk else "rgba(14,116,144,.35)"
_hs_lbl      = "rgba(255,255,255,.30)"         if _dk else "rgba(13,27,46,.36)"
_hm_bg       = "rgba(2,6,14,.97)"             if _dk else "rgba(237,248,255,.98)"
_hm_bdr      = "rgba(56,189,248,.16)"          if _dk else "rgba(14,116,144,.20)"
_hm_hdr      = "rgba(4,10,24,1)"              if _dk else "rgba(220,242,254,1)"
_hm_hdr_b    = "rgba(56,189,248,.10)"          if _dk else "rgba(14,116,144,.14)"
_hm_div      = "rgba(255,255,255,.05)"         if _dk else "rgba(14,116,144,.10)"
_hm_sl       = "rgba(255,255,255,.28)"         if _dk else "rgba(13,27,46,.36)"
_hm_lbl      = "rgba(255,255,255,.36)"         if _dk else "rgba(13,27,46,.44)"
_hm_note     = "rgba(255,255,255,.28)"         if _dk else "rgba(13,27,46,.38)"
_hm_leg      = "rgba(255,255,255,.44)"         if _dk else "rgba(13,27,46,.50)"
_hm_img_b    = "rgba(255,255,255,.08)"         if _dk else "rgba(14,116,144,.14)"
_hm_ex_bg    = "rgba(255,255,255,.018)"        if _dk else "rgba(255,255,255,.58)"
_hm_ex_bdr   = "rgba(255,255,255,.06)"         if _dk else "rgba(14,116,144,.14)"
_hm_ex_b     = "rgba(255,255,255,.44)"         if _dk else "rgba(13,27,46,.58)"
_cs_bg       = "rgba(255,255,255,.022)"        if _dk else "rgba(255,255,255,.58)"
_cs_lbl      = "rgba(255,255,255,.32)"         if _dk else "rgba(13,27,46,.42)"
_pr_a        = "rgba(14,30,70,.82)"            if _dk else "rgba(224,240,254,.92)"
_pr_b        = "rgba(8,20,48,.92)"             if _dk else "rgba(210,234,252,.96)"
_pr_bdr      = "rgba(56,189,248,.20)"          if _dk else "rgba(14,116,144,.22)"
_pr_eye      = "rgba(56,189,248,.62)"          if _dk else "rgba(14,116,144,.75)"
_pr_name     = "#f8fafc"                       if _dk else "#0d1b2e"
_cf_trk      = "rgba(255,255,255,.08)"         if _dk else "rgba(14,116,144,.12)"
_cf_l        = "rgba(255,255,255,.40)"         if _dk else "rgba(13,27,46,.44)"
_sb_bg       = "rgba(4,9,15,.98)"              if _dk else "rgba(240,250,255,.98)"
_sb_bdr      = "rgba(56,189,248,.09)"          if _dk else "rgba(14,116,144,.14)"
_tog_bg      = "rgba(28,38,68,.80)"            if _dk else "rgba(219,240,253,.92)"
_tog_bdr     = "rgba(56,189,248,.42)"          if _dk else "rgba(14,116,144,.44)"
_tog_ico     = "#7dd3fc"                       if _dk else "#0369a1"
_tog_hov     = "rgba(56,189,248,.22)"          if _dk else "rgba(14,116,144,.18)"
_dl_bg       = "rgba(56,189,248,.09)"          if _dk else "rgba(14,116,144,.08)"
_dl_bdr      = "rgba(56,189,248,.24)"          if _dk else "rgba(14,116,144,.22)"
_dl_txt      = "#38bdf8"                       if _dk else "#0e7490"
_dl_hov      = "rgba(56,189,248,.16)"          if _dk else "rgba(14,116,144,.14)"
_prog_trk    = "rgba(255,255,255,.08)"         if _dk else "rgba(14,116,144,.12)"
_succ_bg     = "rgba(34,197,94,.08)"           if _dk else "rgba(220,252,231,.72)"
_succ_bdr    = "rgba(34,197,94,.25)"           if _dk else "rgba(34,197,94,.35)"
_succ_txt    = "#86efac"                       if _dk else "#14532d"
_err_bg      = "rgba(239,68,68,.08)"           if _dk else "rgba(254,226,226,.72)"
_err_bdr     = "rgba(239,68,68,.25)"           if _dk else "rgba(239,68,68,.35)"
_err_txt     = "#fca5a5"                       if _dk else "#7f1d1d"
_warn_bg     = "rgba(245,158,11,.08)"          if _dk else "rgba(254,243,199,.72)"
_warn_bdr    = "rgba(245,158,11,.25)"          if _dk else "rgba(245,158,11,.35)"
_warn_txt    = "#fcd34d"                       if _dk else "#78350f"
_info_bg     = "rgba(56,189,248,.07)"          if _dk else "rgba(224,242,254,.72)"
_info_bdr    = "rgba(56,189,248,.22)"          if _dk else "rgba(14,116,144,.30)"
_info_txt    = "#7dd3fc"                       if _dk else "#0c4a6e"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@300;400;500;600;700&family=DM+Mono:wght@300;400;500&display=swap');

*,*::before,*::after{{box-sizing:border-box}}
html,body,[class*="css"]{{font-family:'Inter',sans-serif;-webkit-font-smoothing:antialiased;}}

.stApp{{
  background:{_bg};
  background-image:
    radial-gradient(ellipse 90% 55% at 10% -5%, {_bg_ga} 0%, transparent 45%),
    radial-gradient(ellipse 70% 45% at 95% 105%, {_bg_gb} 0%, transparent 45%),
    radial-gradient(ellipse 50% 70% at 50% 50%, {_bg_gc} 0%, transparent 70%);
  color:{_txt};min-height:100vh;
}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding:0 !important;max-width:100% !important}}

/* NAV */
.topnav{{
  position:sticky;top:0;z-index:200;
  background:{_nav_bg};
  backdrop-filter:blur(24px) saturate(180%);
  -webkit-backdrop-filter:blur(24px) saturate(180%);
  border-bottom:1px solid {_nav_bdr};
  padding:.7rem 2.2rem;
  display:flex;align-items:center;justify-content:space-between;gap:1rem;
}}
.nav-brand{{display:flex;align-items:center;gap:13px}}
.nav-logo{{
  width:38px;height:38px;border-radius:10px;font-size:18px;
  background:linear-gradient(135deg,#1e40af,#0e7490);
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 20px rgba(56,189,248,.40);flex-shrink:0;
}}
.nav-name{{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:{_txt};letter-spacing:-.3px}}
.nav-name span{{color:#38bdf8}}
.nav-tagline{{font-family:'DM Mono',monospace;font-size:8px;color:{_txt_d};letter-spacing:.15em;text-transform:uppercase;margin-top:1px}}
.nav-right{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}}
.chip{{font-family:'DM Mono',monospace;font-size:9px;font-weight:500;padding:4px 10px;border-radius:20px;letter-spacing:.06em;text-transform:uppercase;white-space:nowrap}}
.c-blue  {{background:rgba(59,130,246,.14);color:#93c5fd;border:1px solid rgba(59,130,246,.28)}}
.c-teal  {{background:rgba(20,184,166,.14);color:#5eead4;border:1px solid rgba(20,184,166,.28)}}
.c-green {{background:rgba(34,197,94,.14); color:#86efac;border:1px solid rgba(34,197,94,.28)}}
.c-amber {{background:rgba(245,158,11,.14);color:#fcd34d;border:1px solid rgba(245,158,11,.28)}}
.c-purple{{background:rgba(139,92,246,.14);color:#c4b5fd;border:1px solid rgba(139,92,246,.28)}}

/* HERO */
.hero{{
  position:relative;overflow:hidden;
  padding:3rem 2.4rem 2.6rem;
  background:linear-gradient(130deg,{_hero_a} 0%,{_hero_b} 55%,{_hero_a} 100%);
  border-bottom:1px solid {_hero_bdr};
}}
.hero::before{{
  content:'';position:absolute;inset:0;pointer-events:none;
  background:
    radial-gradient(ellipse 60% 80% at 82% 45%, rgba(56,189,248,.07) 0%,transparent 55%),
    radial-gradient(ellipse 40% 60% at 18% 75%, rgba(99,102,241,.06) 0%,transparent 50%),
    repeating-linear-gradient(90deg,transparent,transparent 80px,rgba(255,255,255,.012) 80px,rgba(255,255,255,.012) 81px),
    repeating-linear-gradient(0deg,transparent,transparent 80px,rgba(255,255,255,.012) 80px,rgba(255,255,255,.012) 81px);
}}
.hero-inner{{position:relative;z-index:1;max-width:1400px;margin:0 auto}}
.hero-top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}}
.hero-h1{{font-family:'Space Grotesk',sans-serif;font-size:2.55rem;font-weight:700;
  color:{_txt};letter-spacing:-.7px;line-height:1.13;margin-bottom:.5rem}}
.hero-h1 .grad{{background:linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text}}
.hero-desc{{font-size:13.5px;color:{_txt_m};line-height:1.74;max-width:530px}}
.hero-stats{{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end}}
.hs{{text-align:right}}
.hs-n{{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;
  background:linear-gradient(92deg,#38bdf8,#818cf8);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;line-height:1}}
.hs-l{{font-family:'DM Mono',monospace;font-size:9px;color:{_hs_lbl};text-transform:uppercase;letter-spacing:.12em;margin-top:3px}}
.hero-div{{height:1px;margin:1.4rem 0;background:linear-gradient(90deg,rgba(56,189,248,.30),rgba(129,140,248,.18),transparent)}}
.pipeline{{display:flex;align-items:center;gap:0;flex-wrap:wrap}}
.pip-step{{display:flex;align-items:center;gap:9px}}
.pip-num{{width:28px;height:28px;border-radius:50%;
  background:linear-gradient(135deg,#1d4ed8,#0891b2);
  font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#fff;
  display:flex;align-items:center;justify-content:center;
  box-shadow:0 0 14px rgba(56,189,248,.32);flex-shrink:0}}
.pip-txt{{font-size:11.5px;color:{_pip_txt};line-height:1.38}}
.pip-txt strong{{color:{_pip_str};display:block;font-size:11px}}
.pip-arr{{color:{_pip_arr};font-size:20px;padding:0 10px}}

/* WRAP + GLASS */
.wrap{{max-width:1400px;margin:0 auto;padding:2rem 2.2rem 5rem}}
.glass{{
  background:{_glass};
  border:1px solid {_glass_bdr};
  border-radius:20px;padding:1.6rem;
  backdrop-filter:blur(12px);
  box-shadow:0 8px 40px {_glass_shd};
}}
.slbl{{font-family:'DM Mono',monospace;font-size:9.5px;color:{_slbl};
  text-transform:uppercase;letter-spacing:.17em;margin-bottom:12px;
  display:flex;align-items:center;gap:8px}}
.slbl::after{{content:'';flex:1;height:1px;background:{_slbl_ln}}}

/* PREDICTION */
.pred-card{{
  background:linear-gradient(135deg,{_pr_a},{_pr_b});
  border:1px solid {_pr_bdr};
  border-radius:18px;padding:1.5rem 1.6rem;margin-bottom:14px;
  position:relative;overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,.25);
}}
.pred-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8,#818cf8)}}
.pred-eyebrow{{font-family:'DM Mono',monospace;font-size:9px;color:{_pr_eye};text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px}}
.pred-name{{font-family:'Space Grotesk',sans-serif;font-size:38px;font-weight:700;color:{_pr_name};letter-spacing:-1px;line-height:1.04;margin-bottom:15px}}
.conf-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}}
.conf-l{{font-family:'DM Mono',monospace;font-size:10px;color:{_cf_l}}}
.conf-v{{font-family:'Space Grotesk',sans-serif;font-size:14px;color:#38bdf8;font-weight:600}}
.conf-track{{background:{_cf_trk};border-radius:8px;height:6px;overflow:hidden;margin-bottom:15px}}
.conf-fill{{height:100%;border-radius:8px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8)}}
.risk-chip{{display:inline-flex;align-items:center;gap:7px;padding:6px 15px;
  border-radius:20px;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}}
.rdot{{width:6px;height:6px;border-radius:50%}}
.rH{{background:rgba(239,68,68,.13);color:#fca5a5;border:1px solid rgba(239,68,68,.30)}}
.rdH{{background:#ef4444;box-shadow:0 0 7px rgba(239,68,68,.5)}}
.rM{{background:rgba(245,158,11,.13);color:#fcd34d;border:1px solid rgba(245,158,11,.30)}}
.rdM{{background:#f59e0b;box-shadow:0 0 7px rgba(245,158,11,.5)}}
.rL{{background:rgba(34,197,94,.13);color:#86efac;border:1px solid rgba(34,197,94,.30)}}
.rdL{{background:#22c55e;box-shadow:0 0 7px rgba(34,197,94,.5)}}

/* HEATMAP */
.hm-section{{
  background:{_hm_bg};border:1px solid {_hm_bdr};
  border-radius:22px;overflow:hidden;
  box-shadow:0 0 90px rgba(56,189,248,.06),0 28px 70px rgba(0,0,0,.45);
  margin:1.8rem 0;
}}
.hm-header{{
  background:{_hm_hdr};border-bottom:1px solid {_hm_hdr_b};
  padding:1.1rem 1.7rem;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px;
}}
.hm-title{{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:600;color:{_txt};letter-spacing:-.3px}}
.hm-sub{{font-family:'DM Mono',monospace;font-size:9px;color:{_hm_sl};text-transform:uppercase;letter-spacing:.13em;margin-top:3px}}
.hm-legend{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.hm-leg{{display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:9px;color:{_hm_leg}}}
.hm-swatch{{width:28px;height:9px;border-radius:3px}}
.hm-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-bottom:1px solid {_hm_div}}}
.hm-stat{{padding:13px 16px;border-right:1px solid {_hm_div};text-align:center}}
.hm-stat:last-child{{border-right:none}}
.hm-sv{{font-family:'Space Grotesk',sans-serif;font-size:20px;font-weight:600;color:#38bdf8;line-height:1}}
.hm-sl{{font-family:'DM Mono',monospace;font-size:8.5px;color:{_hm_sl};text-transform:uppercase;letter-spacing:.1em;margin-top:4px}}
.hm-grid{{padding:1.4rem 1.6rem}}
.hm-col-lbl{{font-family:'DM Mono',monospace;font-size:9px;color:{_hm_lbl};text-transform:uppercase;letter-spacing:.13em;text-align:center;margin-bottom:8px}}
.hm-col-note{{font-family:'DM Mono',monospace;font-size:9px;color:{_hm_note};text-align:center;margin-top:8px;line-height:1.6}}
.hm-img-frame{{border-radius:12px;overflow:hidden;border:1px solid {_hm_img_b};box-shadow:0 8px 32px rgba(0,0,0,.40)}}
.cscale{{background:{_cs_bg};margin:0 1.6rem 1.4rem;border-radius:8px;padding:9px 13px;border:1px solid {_hm_div}}}
.cscale-bar{{height:12px;border-radius:3px;background:linear-gradient(90deg,#00007f 0%,#0000ff 12%,#007fff 24%,#00ffff 36%,#7fff7f 50%,#ffff00 64%,#ff7f00 76%,#ff0000 88%,#7f0000 100%);margin-bottom:5px}}
.cscale-lbls{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:8.5px;color:{_cs_lbl}}}
.hm-explain{{background:{_hm_ex_bg};border-top:1px solid {_hm_ex_bdr};padding:1.3rem 1.7rem}}
.hm-exp-title{{font-family:'DM Mono',monospace;font-size:9px;color:{_slbl};text-transform:uppercase;letter-spacing:.15em;margin-bottom:12px}}
.hm-exp-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}
.hm-exp-item{{background:{_hm_ex_bg};border:1px solid {_hm_ex_bdr};border-radius:10px;padding:12px 14px}}
.hm-exp-t{{font-family:'DM Mono',monospace;font-size:9px;color:#38bdf8;margin-bottom:5px;font-weight:500}}
.hm-exp-b{{font-size:11.5px;color:{_hm_ex_b};line-height:1.65}}

/* REPORT BLOCKS */
.rb{{border-left:3px solid rgba(147,197,253,.45);background:{_rb_bg};
  border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px}}
.rb-red{{border-left-color:rgba(248,113,113,.60);background:{"rgba(239,68,68,.04)" if _dk else "rgba(254,226,226,.48)"}}}
.rb-yel{{border-left-color:rgba(251,191,36,.60);background:{"rgba(245,158,11,.04)" if _dk else "rgba(254,243,199,.52)"}}}
.rb-grn{{border-left-color:rgba(52,211,153,.60);background:{"rgba(16,185,129,.04)" if _dk else "rgba(209,250,229,.48)"}}}
.rb-t{{font-family:'DM Mono',monospace;font-size:9px;color:{_rb_ttl};text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px}}
.rb-b{{font-size:13px;line-height:1.88;color:{_rb_txt}}}

/* DISCLAIMER */
.disc{{background:{_disc_bg};border:1px solid {_disc_bdr};
  border-left:3px solid rgba(245,158,11,.55);border-radius:0 12px 12px 0;
  padding:13px 18px;font-family:'DM Mono',monospace;font-size:10px;
  color:{_disc_txt};line-height:1.78;margin-top:20px}}
.disc strong{{color:#fbbf24}}

/* STREAMLIT OVERRIDES */
[data-testid="stSidebar"]{{background:{_sb_bg} !important;border-right:1px solid {_sb_bdr} !important}}
[data-testid="stSidebar"] .block-container{{padding:1.5rem 1rem !important}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{{color:{_txt} !important}}
.stTabs [data-baseweb="tab-list"]{{background:{_tab_bg} !important;border-radius:11px !important;padding:3px !important;gap:2px !important;border:1px solid {_tab_bdr} !important}}
.stTabs [data-baseweb="tab"]{{border-radius:8px !important;font-family:'DM Mono',monospace !important;font-size:10.5px !important;color:{_tab_col} !important;padding:8px 15px !important}}
.stTabs [aria-selected="true"]{{background:rgba(56,189,248,.15) !important;color:#38bdf8 !important;box-shadow:none !important}}

/* FILE UPLOADER */
[data-testid="stFileUploader"]{{border:2px dashed {_up_bdr} !important;border-radius:16px !important;background:{_up_bg} !important;}}
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] div{{color:{_up_txt} !important}}
[data-testid="stFileUploader"] button,
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] [data-testid="baseButton-secondary"]{{
  background:linear-gradient(135deg,#1e40af,#0e7490) !important;
  color:#ffffff !important;border:none !important;border-radius:10px !important;
  font-family:'Space Grotesk',sans-serif !important;font-weight:600 !important;
  font-size:13px !important;padding:10px 22px !important;
  box-shadow:0 4px 18px rgba(14,116,144,.45),0 0 0 1px rgba(56,189,248,.22) !important;
  opacity:1 !important;transition:all .22s ease !important;}}
[data-testid="stFileUploader"] button:hover,[data-testid="stFileUploaderDropzone"] button:hover{{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 28px rgba(14,116,144,.6),0 0 0 1px rgba(56,189,248,.38),0 0 30px rgba(56,189,248,.12) !important;
  filter:brightness(1.1) !important;}}

/* ALL BUTTONS */
.stButton > button{{
  background:linear-gradient(135deg,#1e3a8a 0%,#0e7490 55%,#0891b2 100%) !important;
  color:#fff !important;border:none !important;border-radius:13px !important;
  font-family:'Space Grotesk',sans-serif !important;font-weight:600 !important;
  font-size:15px !important;padding:14px 28px !important;width:100% !important;
  box-shadow:0 4px 24px rgba(14,116,144,.4),0 0 0 1px rgba(56,189,248,.12) !important;
  transition:all .22s ease !important;letter-spacing:-.1px !important}}
.stButton > button:hover{{
  transform:translateY(-2px) !important;
  box-shadow:0 8px 32px rgba(14,116,144,.55),0 0 0 1px rgba(56,189,248,.22),0 0 40px rgba(56,189,248,.1) !important}}
.stButton > button:active{{transform:translateY(0px) !important}}

/* THEME TOGGLE — pure HTML button inside nav, no Streamlit column */
.theme-toggle{{
  width:34px;height:34px;border-radius:50%;
  background:{_tog_bg};
  border:1px solid {_tog_bdr};
  color:{_tog_ico};
  font-size:16px;line-height:1;
  display:inline-flex;align-items:center;justify-content:center;
  cursor:pointer;
  box-shadow:0 2px 10px rgba(0,0,0,.20);
  transition:background .18s,transform .18s,box-shadow .18s;
  flex-shrink:0;
  -webkit-appearance:none;appearance:none;
  outline:none;
  text-decoration:none;
}}
.theme-toggle:hover{{
  background:{_tog_hov};
  transform:scale(1.12) rotate(14deg);
  box-shadow:0 4px 18px rgba(56,189,248,.28);
}}

[data-testid="stSelectbox"] > div > div{{background:{_sel_bg} !important;border:1px solid {_sel_bdr} !important;border-radius:11px !important;color:{_sel_txt} !important}}
[data-testid="stMetric"]{{background:{_met_bg} !important;border:1px solid {_met_bdr} !important;border-radius:14px !important;padding:13px 16px !important}}
[data-testid="stMetricLabel"]{{color:{_met_lbl} !important;font-size:11px !important}}
[data-testid="stMetricValue"]{{font-family:'Space Grotesk',sans-serif !important;color:#38bdf8 !important;font-size:22px !important}}
[data-testid="stProgress"] > div{{background:{_prog_trk} !important;border-radius:4px !important}}
[data-testid="stProgress"] > div > div{{background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) !important;border-radius:4px !important}}
[data-testid="stToggle"] label{{color:{_txt_m} !important;font-size:13px !important}}
[data-testid="stSlider"] > div > div > div{{background:#1d4ed8 !important}}
[data-testid="stDownloadButton"] > button{{background:{_dl_bg} !important;border:1px solid {_dl_bdr} !important;color:{_dl_txt} !important;font-family:'DM Mono',monospace !important;font-size:11.5px !important;border-radius:9px !important;padding:9px 16px !important;width:100% !important;box-shadow:none !important}}
[data-testid="stDownloadButton"] > button:hover{{background:{_dl_hov} !important;transform:none !important}}
code{{font-family:'DM Mono',monospace !important;font-size:11px !important;background:{_code_bg} !important;color:{_code_txt} !important;border:1px solid {_code_bdr} !important;border-radius:4px !important}}
hr{{border-color:{_hr} !important;margin:1.8rem 0 !important}}
[data-testid="stImage"] img{{border-radius:12px !important;border:1px solid {_img_bdr} !important;display:block !important}}
div[data-testid="stSuccess"]{{background:{_succ_bg} !important;border-color:{_succ_bdr} !important;color:{_succ_txt} !important}}
div[data-testid="stError"]  {{background:{_err_bg}  !important;border-color:{_err_bdr}  !important;color:{_err_txt}  !important}}
div[data-testid="stWarning"]{{background:{_warn_bg} !important;border-color:{_warn_bdr} !important;color:{_warn_txt} !important}}
div[data-testid="stInfo"]   {{background:{_info_bg} !important;border-color:{_info_bdr} !important;color:{_info_txt} !important}}
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
# _dk is already resolved at the top (query_params checked before CSS)
_next_theme   = "light" if _dk else "dark"
_toggle_icon  = "☀️"   if _dk else "🌙"
_toggle_title = "Switch to light mode" if _dk else "Switch to dark mode"

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
    <a href="?theme={_next_theme}"
       class="theme-toggle"
       title="{_toggle_title}"
       aria-label="{_toggle_title}">{_toggle_icon}</a>
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
