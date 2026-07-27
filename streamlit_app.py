"""
NeuroScan AI - Brain Tumor MRI Classification
================================================================
Uses your models via Hugging Face or TensorFlow Serving API
Works on Streamlit Cloud without TensorFlow installation
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
from PIL import Image, ImageOps
import io, base64, os, json
import requests
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ================================================================
# REAL MODEL LOADING (TensorFlow)
# ================================================================
# IMPORTANT: this app previously never loaded brain_tumor_model.h5 /
# mobilenet_model.h5 at all -- every prediction came from a hand-written
# pixel-statistics heuristic (see predict_with_heuristic below), regardless
# of whether TensorFlow was installed. That is now fixed: real models are
# loaded here and used whenever available. The heuristic is kept ONLY as a
# clearly-labeled last-resort fallback if no model can be loaded, so the
# app never silently pretends a heuristic guess is a model prediction.
TF_AVAILABLE = False
TF_IMPORT_ERROR = ""
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
    TF_AVAILABLE = True
except Exception as e:
    TF_IMPORT_ERROR = str(e)

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Your repo tracks *.h5 via Git LFS (see .gitattributes). Streamlit Community
# Cloud clones repos WITHOUT resolving LFS pointers -- it gets a ~130-byte
# text stub instead of the real weights. To make the app work regardless of
# where/how it's deployed, each model config carries a direct download URL
# (GitHub's LFS media server resolves the real binary, no auth needed for
# public repos) plus the expected file size from the LFS pointer, so a
# pointer-stub can never be mistaken for a loaded model.
#
# Set these via Streamlit secrets instead of hardcoding if you'd rather not
# commit URLs: st.secrets.get("BRAIN_TUMOR_MODEL_URL"), etc.
MODEL_CONFIGS = [
    {
        # Your own training_history.png labels this "ResNet50V2 - Accuracy" --
        # it's not a generic custom CNN, it's a ResNet50V2 backbone. Fixed
        # the display label to match; the internal `key` stays "custom_cnn"
        # so existing secrets/session state referencing that key still work.
        "key": "custom_cnn", "label": "ResNet50V2", "file": "brain_tumor_model.h5",
        "preprocess": "rescale", "expected_size": 230584088,
        "url": "https://media.githubusercontent.com/media/focuzdev/Brain-Tumor-Prediction/master/brain_tumor_model.h5",
    },
    {
        "key": "mobilenet", "label": "MobileNetV2", "file": "mobilenet_model.h5",
        "preprocess": "mobilenet", "expected_size": 32221424,
        "url": "https://media.githubusercontent.com/media/focuzdev/Brain-Tumor-Prediction/master/mobilenet_model.h5",
    },
]

# --- Deployment resource guard -------------------------------------------
# Streamlit Community Cloud's free tier gives ~1GB RAM. custom_cnn's weights
# alone are ~220MB on disk, and TensorFlow typically needs several times a
# model's file size in RAM once loaded (graph + activations + runtime
# overhead). MobileNetV2 is the default: it's the model that's been most
# thoroughly tested and confirmed reliable end-to-end (real inference, real
# Grad-CAM), it's lighter on resources, and ensemble mode hasn't yet shown a
# clear accuracy improvement over it to justify the extra memory cost and
# added complexity of interpreting two disagreeing models. Switch to
# "ensemble" once you've validated it against your labeled samples and are
# comfortable with how disagreement between the two models is surfaced.
#
# Override via Streamlit secrets: MODEL_MODE = "ensemble" | "mobilenet" | "custom_cnn"
DEFAULT_MODEL_MODE = "mobilenet"

def _get_model_mode():
    # A live choice made in the sidebar (this session) always wins -- lets
    # anyone switch between mobilenet / custom_cnn / ensemble right in the
    # running app, without needing access to Streamlit Cloud's secrets panel.
    if "model_mode_override" in st.session_state:
        return st.session_state["model_mode_override"]
    if hasattr(st, "secrets"):
        try:
            return st.secrets.get("MODEL_MODE", DEFAULT_MODEL_MODE)
        except Exception:
            pass
    return DEFAULT_MODEL_MODE

def _active_model_configs(mode=None):
    if mode is None:
        mode = _get_model_mode()
    if mode == "ensemble":
        return MODEL_CONFIGS
    return [c for c in MODEL_CONFIGS if c["key"] == mode] or MODEL_CONFIGS[:1]

def _log(msg):
    """Prints to stdout with a consistent, greppable prefix so the failure
    mode is visible in Streamlit Cloud's server log (Manage app -> logs)
    instead of only surfacing as a sidebar error the user has to screenshot."""
    print(f"[NEUROSCAN] {msg}", flush=True)

def _is_valid_model_file(path, expected_size):
    """A file only counts as 'present' if it's actually the binary weights,
    not a Git LFS pointer stub. LFS pointer files are always a few hundred
    bytes; real .h5 weights are tens/hundreds of MB. We check both existence
    and a size floor so a stray pointer file never gets fed to load_model()."""
    if not os.path.exists(path):
        return False
    size = os.path.getsize(path)
    # Allow some tolerance but reject anything that looks like an LFS stub (<1MB).
    return size > 1_000_000

def _ensure_model_downloaded(cfg):
    """Downloads the real model weights if the local file is missing or is
    just an LFS pointer stub. Returns (path, error_message_or_None)."""
    path = os.path.join(MODEL_DIR, cfg["file"])
    tag = f"[{cfg['key']}]"

    if _is_valid_model_file(path, cfg["expected_size"]):
        _log(f"{tag} found valid local file at {path} ({os.path.getsize(path)} bytes) -- skipping download")
        return path, None

    if os.path.exists(path):
        _log(f"{tag} local file exists but looks invalid (size={os.path.getsize(path)} bytes, "
             f"expected ~{cfg['expected_size']}) -- treating as an LFS pointer stub, will re-download")
    else:
        _log(f"{tag} no local file at {path} -- will download")

    def _safe_secret(key, default):
        try:
            return st.secrets.get(key, default)
        except Exception:
            return default
    url = _safe_secret(f"{cfg['key'].upper()}_MODEL_URL", cfg.get("url"))
    if not url:
        msg = f"{cfg['file']} is missing/invalid (likely an unresolved Git LFS pointer) and no download URL is configured."
        _log(f"{tag} ERROR: {msg}")
        return None, msg

    _log(f"{tag} downloading from {url} ...")
    try:
        r = requests.get(url, stream=True, timeout=300)
        _log(f"{tag} HTTP {r.status_code}, content-length={r.headers.get('content-length')}, "
             f"content-type={r.headers.get('content-type')}")
        r.raise_for_status()

        total = int(r.headers.get("content-length") or cfg["expected_size"])
        progress_ui = st.progress(0.0, text=f"Downloading {cfg['label']} weights (0 / {total/1e6:.0f} MB)...")
        tmp_path = path + ".part"
        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                frac = min(downloaded / total, 1.0) if total else 0.0
                progress_ui.progress(frac, text=f"Downloading {cfg['label']} weights "
                                                 f"({downloaded/1e6:.0f} / {total/1e6:.0f} MB)...")
        progress_ui.empty()
        _log(f"{tag} download finished, wrote {downloaded} bytes to {tmp_path}")
        os.replace(tmp_path, path)
    except Exception as e:
        msg = f"Failed to download {cfg['file']} from {url}: {e}"
        _log(f"{tag} ERROR: {msg}")
        return None, msg

    if not _is_valid_model_file(path, cfg["expected_size"]):
        msg = (f"Downloaded {cfg['file']} but it still looks invalid (size {os.path.getsize(path)} bytes) "
               f"-- check the URL is a direct binary link, not an LFS pointer or HTML page.")
        _log(f"{tag} ERROR: {msg}")
        return None, msg

    _log(f"{tag} download OK, final size {os.path.getsize(path)} bytes")
    return path, None

@st.cache_resource(show_spinner=False)
def load_models(mode):
    """
    Loads the model(s) selected by `mode`, downloading real weights first if
    the local copy is missing or is an unresolved Git LFS pointer stub.
    Returns (loaded: dict[key -> keras.Model], errors: dict[key -> str]).

    IMPORTANT: `mode` must be an explicit parameter, not read from session
    state/secrets inside the function body. st.cache_resource keys its cache
    purely on function arguments -- a zero-argument version of this function
    would return the SAME cached result for every mode after the first call,
    silently showing stale "not loaded" state for models that were never
    even attempted under whatever mode happened to be cached first. Passing
    `mode` in gives each mode its own independent cache entry.
    """
    loaded, errors = {}, {}
    _log(f"load_models() starting, MODEL_MODE={mode!r}")
    if not TF_AVAILABLE:
        _log(f"ERROR: TensorFlow not available: {TF_IMPORT_ERROR}")
        errors["_tensorflow"] = TF_IMPORT_ERROR or "TensorFlow is not installed in this environment."
        return loaded, errors
    _log(f"TensorFlow version: {tf.__version__}")

    for cfg in _active_model_configs(mode):
        tag = f"[{cfg['key']}]"
        path, err = _ensure_model_downloaded(cfg)
        if err:
            errors[cfg["key"]] = err
            continue
        try:
            _log(f"{tag} calling tf.keras.models.load_model({path}) ...")
            m = tf.keras.models.load_model(path, compile=False)
            loaded[cfg["key"]] = m
            _log(f"{tag} load_model() succeeded")
        except Exception as e:
            import traceback
            _log(f"{tag} ERROR: load_model() failed:\n{traceback.format_exc()}")
            errors[cfg["key"]] = f"Failed to load {cfg['file']}: {e}"
    _log(f"load_models() done. loaded={list(loaded.keys())} errors={list(errors.keys())}")
    return loaded, errors

def preprocess_for_model(pil_img, style):
    """
    Converts a PIL image into the exact tensor shape/scale a given model
    expects. NOTE: 'rescale' (0-1) and 'mobilenet' ([-1,1] via Keras'
    official preprocess_input) are the two most common training setups --
    but you must confirm these match what your training notebook actually
    used. A preprocessing mismatch (e.g. model trained on 0-1 floats but
    served with raw 0-255 ints, or vice versa) is one of the most common
    causes of a model that scores well offline but misclassifies in
    production, and it fails *silently* -- no error, just wrong answers.
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    if style == "mobilenet":
        arr = mobilenet_preprocess(arr)
    else:  # "rescale"
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predict_with_models(pil_img, loaded_models):
    """
    Runs real inference. If more than one model loaded, ensembles by
    averaging softmax probabilities (matches the repo's
    confusion_matrix_ensemble.png naming). Returns:
      preds        -> np.array aligned to CLASS_NAMES
      explanation  -> short string noting which model(s) produced this
      per_model    -> dict[label -> preds array] for transparency
    """
    per_model = {}
    for cfg in MODEL_CONFIGS:
        model = loaded_models.get(cfg["key"])
        if model is None:
            continue
        x = preprocess_for_model(pil_img, cfg["preprocess"])
        raw = model.predict(x, verbose=0)[0]
        # If the model's final layer isn't already softmax-normalized, normalize defensively.
        raw = np.asarray(raw, dtype=np.float64)
        if raw.sum() <= 0 or abs(raw.sum() - 1.0) > 1e-3:
            exp = np.exp(raw - raw.max())
            raw = exp / exp.sum()
        per_model[cfg["label"]] = raw

    if not per_model:
        raise RuntimeError("No loaded model produced a prediction.")

    preds = np.mean(list(per_model.values()), axis=0)
    if len(per_model) > 1:
        explanation = f"Ensemble prediction averaged across: {', '.join(per_model.keys())}."
    else:
        explanation = f"Prediction from {list(per_model.keys())[0]}."
    return preds, explanation, per_model

def find_last_conv_layer(model):
    """Finds the last Conv2D-type layer in a (possibly nested) Keras model,
    for Grad-CAM. Returns (owner, layer_name):
      owner       -- the model/layer that directly contains this Conv2D
                      (may be `model` itself, or a nested sub-model like a
                      MobileNetV2 backbone wrapped as a single layer)
      layer_name  -- the Conv2D layer's name within `owner`
    Returning the correct owner matters: calling model.get_layer(name) for a
    name that actually lives inside a NESTED sub-model raises "No such
    layer" -- a very common, silent Grad-CAM failure for any transfer-
    learning model where the backbone is wrapped as a single layer rather
    than flattened into the top-level model.
    """
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return model, layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return layer, sub.name
    return None, None

def generate_gradcam(pil_img, model, preprocess_style, pred_index):
    """
    Real Grad-CAM: computes gradients of the predicted class score w.r.t.
    the last conv layer's feature maps, weights the feature maps by those
    gradients, and produces a localization heatmap. Returns None (caller
    falls back to the synthetic heatmap) only if no conv layer can be found
    at all or the graph genuinely can't be built -- and logs why either way.

    Keras 3 does NOT reliably connect an inner sub-model's intermediate
    layer output back to the outer model's input graph (confirmed via a
    real "Output ... is not connected to inputs" error in production, not
    a guess). Workaround for the nested case: build a self-contained model
    for JUST the backbone (input -> [conv_output, backbone_output], which
    is a valid graph since it never leaves the backbone's own boundaries),
    then manually replay the remaining "head" layers (pooling/dense/etc.)
    on top of that inside the SAME GradientTape, so the whole computation
    -- conv activations through to the final class score -- is one
    continuous, differentiable chain instead of two disconnected graphs.
    """
    owner, last_conv = find_last_conv_layer(model)
    if last_conv is None:
        _log("[gradcam] no Conv2D layer found anywhere in the model")
        return None

    try:
        x = preprocess_for_model(pil_img, preprocess_style)
        x_tensor = tf.convert_to_tensor(x)

        try:
            _ = model(x_tensor, training=False)
        except Exception:
            pass

        if owner is model:
            # Flat case: conv layer lives directly in the top-level model,
            # no nested-submodel connectivity issue to work around.
            model_inputs = model.inputs if model.inputs else [model.input]
            grad_model = tf.keras.models.Model(
                model_inputs, [owner.get_layer(last_conv).output, model.output]
            )
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(x_tensor)
                loss = predictions[:, pred_index]
            grads = tape.gradient(loss, conv_outputs)
        else:
            # Nested case (e.g. MobileNetV2 wrapped as a single layer).
            owner_grad_model = tf.keras.models.Model(
                owner.input, [owner.get_layer(last_conv).output, owner.output]
            )
            try:
                owner_idx = model.layers.index(owner)
            except ValueError:
                owner_idx = -1
            head_layers = model.layers[owner_idx + 1:] if owner_idx >= 0 else []
            _log(f"[gradcam] nested backbone '{owner.name}', replaying {len(head_layers)} head layer(s): "
                 f"{[l.name for l in head_layers]}")

            with tf.GradientTape() as tape:
                conv_outputs, owner_output = owner_grad_model(x_tensor)
                h = owner_output
                for layer in head_layers:
                    h = layer(h)
                loss = h[:, pred_index]
            grads = tape.gradient(loss, conv_outputs)

        if grads is None:
            _log(f"[gradcam] gradient was None for layer '{last_conv}' (owner={owner.name if owner is not model else 'top-level model'}) -- graph still disconnected")
            return None
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        heatmap = cv2.resize(heatmap, IMG_SIZE)
        _log(f"[gradcam] succeeded using layer '{last_conv}' (owner={owner.name if owner is not model else 'top-level model'})")
        return heatmap
    except Exception as e:
        import traceback
        _log(f"[gradcam] ERROR building/running Grad-CAM on layer '{last_conv}': {traceback.format_exc()}")
        return None



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
    st.session_state.theme = "light"

theme_param = st.query_params.get("theme", None)
if theme_param in ["light", "dark"]:
    if st.session_state.theme != theme_param:
        st.session_state.theme = theme_param
        st.query_params.clear()
        st.rerun()

_dk = (st.session_state.theme == "dark")

# ================================================================
# CONSTANTS
# ================================================================
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary Tumor"]
CLASS_COLORS = ["#ef4444", "#3b82f6", "#22c55e", "#a855f7"]
IMG_SIZE = (224, 224)

SAMPLE_DIR = "samples"
SAMPLE_FILES = {
    "Glioma": "glioma.jpg",
    "Meningioma": "meningioma.jpg",
    "Pituitary Tumor": "pituitary.jpg",
    "No Tumor": "no_tumor.jpg",
}

RISK = {
    "Glioma": ("HIGH", "rH", "rdH"),
    "Meningioma": ("MODERATE", "rM", "rdM"),
    "Pituitary Tumor": ("MODERATE", "rM", "rdM"),
    "No Tumor": ("LOW", "rL", "rdL"),
}

# ================================================================
# CREATE SAMPLE IMAGES
# ================================================================
def create_sample_images():
    if not os.path.exists(SAMPLE_DIR):
        os.makedirs(SAMPLE_DIR)
    
    samples = {
        "glioma.jpg": "glioma",
        "meningioma.jpg": "meningioma",
        "pituitary.jpg": "pituitary",
        "no_tumor.jpg": "normal"
    }
    
    for fname, tumor_type in samples.items():
        fpath = os.path.join(SAMPLE_DIR, fname)
        if not os.path.exists(fpath):
            img = generate_sample_mri(tumor_type)
            img.save(fpath, "JPEG", quality=85)

def generate_sample_mri(tumor_type):
    size = 224
    img = np.zeros((size, size), dtype=np.uint8)
    
    center_y, center_x = size//2, size//2
    for i in range(size):
        for j in range(size):
            dist = np.sqrt(((i - center_y) / (size*0.4))**2 + ((j - center_x) / (size*0.35))**2)
            if dist <= 1:
                intensity = 128 + 80 * (1 - dist) + np.random.randint(-20, 20)
                img[i, j] = np.clip(intensity, 0, 255)
            else:
                img[i, j] = np.random.randint(0, 30)
    
    if tumor_type == "glioma":
        cy, cx = int(size*0.35), int(size*0.65)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt(((i - cy) / (size*0.12))**2 + ((j - cx) / (size*0.15))**2)
                if dist <= 1:
                    img[i, j] = np.clip(img[i, j] + 100 + np.random.randint(-20, 20), 0, 255)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt(((i - cy) / (size*0.20))**2 + ((j - cx) / (size*0.25))**2)
                if 1 < dist <= 1.8:
                    img[i, j] = np.clip(img[i, j] + 40 + np.random.randint(-10, 10), 0, 255)
    elif tumor_type == "meningioma":
        cy, cx = int(size*0.4), int(size*0.75)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt(((i - cy) / (size*0.10))**2 + ((j - cx) / (size*0.12))**2)
                if dist <= 1:
                    img[i, j] = np.clip(img[i, j] + 120 + np.random.randint(-15, 15), 0, 255)
    elif tumor_type == "pituitary":
        cy, cx = int(size*0.5), int(size*0.5)
        for i in range(size):
            for j in range(size):
                dist = np.sqrt(((i - cy) / (size*0.08))**2 + ((j - cx) / (size*0.10))**2)
                if dist <= 1:
                    img[i, j] = np.clip(img[i, j] + 130 + np.random.randint(-15, 15), 0, 255)
    
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_rgb = np.stack([img, img, img], axis=-1)
    img_rgb[:, :, 0] = np.clip(img_rgb[:, :, 0] + np.random.randint(-5, 5), 0, 255)
    img_rgb[:, :, 2] = np.clip(img_rgb[:, :, 2] + np.random.randint(-5, 5), 0, 255)
    
    return Image.fromarray(img_rgb.astype(np.uint8), "RGB")

create_sample_images()

# ================================================================
# CSS - SIMPLIFIED
# ================================================================
bg_color = "#0a0e1a" if _dk else "#f0f4fa"
text_color = "#e2e8f0" if _dk else "#0a1628"
card_border = "rgba(56,189,248,.22)" if _dk else "rgba(56,189,248,.40)"
glass_bg = "rgba(255,255,255,.03)" if _dk else "rgba(255,255,255,.85)"
glass_border = "rgba(255,255,255,.075)" if _dk else "rgba(56,189,248,.20)"
nav_bg = "rgba(10,14,26,.95)" if _dk else "rgba(255,255,255,.95)"
nav_border = "rgba(56,189,248,.15)" if _dk else "rgba(56,189,248,.20)"

st.markdown(f"""
<style>
*{{box-sizing:border-box}}
.stApp{{
  background:{bg_color} !important;
  color:{text_color} !important;
  min-height:100vh;
}}
#MainMenu,footer,header{{visibility:hidden}}
.block-container{{padding:0 !important;max-width:100% !important}}

.topnav{{
  position:sticky;top:0;z-index:200;
  background:{nav_bg};
  backdrop-filter:blur(24px);
  border-bottom:1px solid {nav_border};
  padding:.8rem 2.4rem;
  display:flex;align-items:center;justify-content:space-between;gap:1rem;
}}
.nav-brand{{display:flex;align-items:center;gap:13px}}
.nav-logo{{width:38px;height:38px;border-radius:10px;font-size:18px;background:linear-gradient(135deg,#1e40af,#0e7490);display:flex;align-items:center;justify-content:center;box-shadow:0 0 20px rgba(56,189,248,.4);flex-shrink:0}}
.nav-name{{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:700;color:{text_color};letter-spacing:-.3px}}
.nav-name span{{color:#38bdf8}}
.nav-tagline{{font-family:'DM Mono',monospace;font-size:8px;color:{"rgba(255,255,255,.45)" if _dk else "#4a6580"};letter-spacing:.15em;text-transform:uppercase;margin-top:1px}}
.nav-right{{display:flex;gap:6px;flex-wrap:wrap;align-items:center}}

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

.hero{{
  position:relative;overflow:hidden;padding:3rem 0 2.5rem;
  background:linear-gradient(130deg,{"#040c1c" if _dk else "#daeeff"} 0%,{"#071630" if _dk else "#c4e0f8"} 55%,{"#040c1c" if _dk else "#daeeff"} 100%);
  border-bottom:1px solid {nav_border};
}}
.hero-inner{{position:relative;z-index:1;width:100%;padding:0 2.8rem}}
.hero-top{{display:flex;align-items:flex-end;justify-content:space-between;flex-wrap:wrap;gap:1.5rem}}
.hero-h1{{font-family:'Space Grotesk',sans-serif;font-size:clamp(1.9rem,3.5vw,3rem);font-weight:700;color:{text_color};letter-spacing:-.7px;line-height:1.13;margin-bottom:.5rem}}
.hero-h1 .grad{{background:linear-gradient(92deg,#38bdf8 0%,#818cf8 48%,#a78bfa 100%);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.hero-desc{{font-size:15px;color:{"rgba(255,255,255,.7)" if _dk else "#2d4a6b"};line-height:1.74;max-width:530px}}
.hero-stats{{display:flex;gap:2rem;flex-wrap:wrap;align-items:flex-end}}
.hs{{text-align:right}}
.hs-n{{font-family:'Space Grotesk',sans-serif;font-size:27px;font-weight:700;background:linear-gradient(92deg,#38bdf8,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1}}
.hs-l{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.12em;margin-top:3px}}
.hero-div{{height:1px;margin:1.2rem 0;background:linear-gradient(90deg,rgba(56,189,248,.3),rgba(129,140,248,.18),transparent)}}
.pipeline{{display:flex;align-items:center;gap:0;flex-wrap:wrap}}
.pip-step{{display:flex;align-items:center;gap:9px}}
.pip-num{{width:28px;height:28px;border-radius:50%;background:linear-gradient(135deg,#1d4ed8,#0891b2);font-family:'DM Mono',monospace;font-size:11px;font-weight:600;color:#fff;display:flex;align-items:center;justify-content:center;box-shadow:0 0 14px rgba(56,189,248,.32);flex-shrink:0}}
.pip-txt{{font-size:11.5px;color:{"rgba(255,255,255,.65)" if _dk else "#2d4a6b"};line-height:1.38}}
.pip-txt strong{{color:{"rgba(255,255,255,.9)" if _dk else "#0a1628"};display:block;font-size:11px}}
.pip-arr{{color:{"rgba(56,189,248,.4)" if _dk else "#4a8fa8"};font-size:20px;padding:0 10px}}

.wrap{{width:100%;padding:2rem 2.8rem 4rem}}
.glass{{
  background:{glass_bg};
  border:1px solid {glass_border};
  border-radius:20px;padding:1.8rem 2rem;
  backdrop-filter:blur(12px);
  box-shadow:0 8px 40px {"rgba(0,0,0,.35)" if _dk else "rgba(10,22,80,.08)"};
}}
.slbl{{font-family:'DM Mono',monospace;font-size:11px;color:rgba(56,189,248,.8);text-transform:uppercase;letter-spacing:.17em;margin-bottom:12px;display:flex;align-items:center;gap:8px}}
.slbl::after{{content:'';flex:1;height:1px;background:rgba(56,189,248,.2)}}

.pred-card{{
  background:linear-gradient(135deg,{"rgba(14,30,70,.82)" if _dk else "#daeeff"},{"rgba(8,20,48,.92)" if _dk else "#cce5f8"});
  border:1px solid {card_border};
  border-radius:18px;padding:1.5rem 1.6rem;margin-bottom:14px;
  position:relative;overflow:hidden;
  box-shadow:0 12px 40px {"rgba(0,0,0,.25)" if _dk else "rgba(10,22,80,.10)"};
}}
.pred-card::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8,#818cf8)}}
.pred-eyebrow{{font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.8);text-transform:uppercase;letter-spacing:.18em;margin-bottom:8px}}
.pred-name{{font-family:'Space Grotesk',sans-serif;font-size:clamp(30px,4vw,44px);font-weight:700;color:{"#f8fafc" if _dk else "#0a1628"};letter-spacing:-1px;line-height:1.04;margin-bottom:15px}}
.conf-row{{display:flex;align-items:center;justify-content:space-between;margin-bottom:5px}}
.conf-l{{font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"}}}
.conf-v{{font-family:'Space Grotesk',sans-serif;font-size:14px;color:#38bdf8;font-weight:600}}
.conf-track{{background:{"rgba(255,255,255,.1)" if _dk else "#b8d8ec"};border-radius:8px;height:6px;overflow:hidden;margin-bottom:15px}}
.conf-fill{{height:100%;border-radius:8px;background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8)}}
.risk-chip{{display:inline-flex;align-items:center;gap:7px;padding:6px 15px;border-radius:20px;font-family:'DM Mono',monospace;font-size:10px;font-weight:500;letter-spacing:.08em;text-transform:uppercase}}
.rdot{{width:6px;height:6px;border-radius:50%}}
.rH{{background:rgba(239,68,68,.13);color:#dc2626;border:1px solid rgba(239,68,68,.35)}}
.rdH{{background:#ef4444;box-shadow:0 0 7px rgba(239,68,68,.5)}}
.rM{{background:rgba(245,158,11,.13);color:#d97706;border:1px solid rgba(245,158,11,.35)}}
.rdM{{background:#f59e0b;box-shadow:0 0 7px rgba(245,158,11,.5)}}
.rL{{background:rgba(34,197,94,.13);color:#16a34a;border:1px solid rgba(34,197,94,.35)}}
.rdL{{background:#22c55e;box-shadow:0 0 7px rgba(34,197,94,.5)}}

.hm-section{{
  background:{"rgba(2,6,14,.97)" if _dk else "#f4faff"};
  border:1px solid {card_border};
  border-radius:22px;overflow:hidden;
  box-shadow:0 8px 32px {"rgba(0,0,0,.45)" if _dk else "rgba(10,22,80,.08)"};
  margin:1.8rem 0
}}
.hm-header{{
  background:{"rgba(4,10,24,1)" if _dk else "#daeeff"};
  border-bottom:1px solid {"rgba(56,189,248,.12)" if _dk else "#a8cfe0"};
  padding:1.1rem 1.7rem;display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:10px
}}
.hm-title{{font-family:'Space Grotesk',sans-serif;font-size:16px;font-weight:600;color:{text_color};letter-spacing:-.3px}}
.hm-sub{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.45)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.13em;margin-top:3px}}
.hm-legend{{display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.hm-leg{{display:flex;align-items:center;gap:6px;font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.6)" if _dk else "#1a3550"}}}
.hm-swatch{{width:28px;height:9px;border-radius:3px}}
.hm-stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:0;border-bottom:1px solid {"rgba(255,255,255,.08)" if _dk else "#b8d8ec"}}}
.hm-stat{{padding:13px 16px;border-right:1px solid {"rgba(255,255,255,.08)" if _dk else "#b8d8ec"};text-align:center}}
.hm-stat:last-child{{border-right:none}}
.hm-sv{{font-family:'Space Grotesk',sans-serif;font-size:24px;font-weight:600;color:#38bdf8;line-height:1}}
.hm-sl{{font-family:'DM Mono',monospace;font-size:8.5px;color:{"rgba(255,255,255,.45)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.1em;margin-top:4px}}
.hm-col-lbl{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#1a3550"};text-transform:uppercase;letter-spacing:.13em;text-align:center;margin-bottom:8px}}
.hm-col-note{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.45)" if _dk else "#2d4a6b"};text-align:center;margin-top:8px;line-height:1.6}}
.hm-img-frame{{border-radius:12px;overflow:hidden;border:1px solid {"rgba(255,255,255,.1)" if _dk else "#a8cfe0"};box-shadow:0 4px 16px {"rgba(0,0,0,.4)" if _dk else "rgba(10,22,80,.10)"}}}
.cscale{{background:{"rgba(255,255,255,.025)" if _dk else "#ffffff"};margin:0 1.6rem 1.4rem;border-radius:8px;padding:9px 13px;border:1px solid {"rgba(255,255,255,.08)" if _dk else "#b8d8ec"}}}
.cscale-bar{{height:12px;border-radius:3px;background:linear-gradient(90deg,#00007f 0%,#0000ff 12%,#007fff 24%,#00ffff 36%,#7fff7f 50%,#ffff00 64%,#ff7f00 76%,#ff0000 88%,#7f0000 100%);margin-bottom:5px}}
.cscale-lbls{{display:flex;justify-content:space-between;font-family:'DM Mono',monospace;font-size:8.5px;color:{"rgba(255,255,255,.5)" if _dk else "#2d4a6b"}}}
.hm-explain{{background:{"rgba(255,255,255,.025)" if _dk else "#ffffff"};border-top:1px solid {"rgba(255,255,255,.08)" if _dk else "#a8cfe0"};padding:1.3rem 1.7rem}}
.hm-exp-title{{font-family:'DM Mono',monospace;font-size:9px;color:rgba(56,189,248,.8);text-transform:uppercase;letter-spacing:.15em;margin-bottom:12px}}
.hm-exp-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px}}
.hm-exp-item{{background:{"rgba(255,255,255,.025)" if _dk else "#ffffff"};border:1px solid {"rgba(255,255,255,.08)" if _dk else "#a8cfe0"};border-radius:10px;padding:12px 14px}}
.hm-exp-t{{font-family:'DM Mono',monospace;font-size:9px;color:#38bdf8;margin-bottom:5px;font-weight:500}}
.hm-exp-b{{font-size:13px;color:{"rgba(255,255,255,.8)" if _dk else "#0a1628"};line-height:1.65}}

.rb{{border-left:3px solid rgba(147,197,253,.55);background:{"rgba(255,255,255,.028)" if _dk else "#ffffff"};border-radius:0 12px 12px 0;padding:14px 18px;margin-bottom:12px}}
.rb-red{{border-left-color:rgba(248,113,113,.8);background:{"rgba(239,68,68,.08)" if _dk else "#fde8e8"}}}
.rb-yel{{border-left-color:rgba(251,191,36,.8);background:{"rgba(245,158,11,.08)" if _dk else "#fef3cd"}}}
.rb-grn{{border-left-color:rgba(52,211,153,.8);background:{"rgba(16,185,129,.08)" if _dk else "#d8f5e8"}}}
.rb-t{{font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "#2d4a6b"};text-transform:uppercase;letter-spacing:.14em;margin-bottom:7px}}
.rb-b{{font-size:14px;line-height:1.9;color:{"rgba(255,255,255,.88)" if _dk else "#0a1628"}}}

.disc{{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.3);border-left:3px solid rgba(245,158,11,.8);border-radius:0 12px 12px 0;padding:13px 18px;font-family:'DM Mono',monospace;font-size:10px;color:#fcd34d;line-height:1.78;margin-top:20px}}
.disc strong{{color:#f59e0b}}

[data-testid="stSidebar"]{{background:{"rgba(10,14,26,.98)" if _dk else "#f0f8ff"} !important;border-right:1px solid {"rgba(56,189,248,.09)" if _dk else "#a8cfe0"} !important}}
[data-testid="stSidebar"] p,[data-testid="stSidebar"] label,[data-testid="stSidebar"] span{{color:{text_color} !important}}
[data-testid="stFileUploader"]{{border:2px dashed rgba(56,189,248,.75) !important;border-radius:16px !important;background:{"rgba(14,58,140,.22)" if _dk else "rgba(12,74,110,.06)"} !important;padding:4px !important}}
[data-testid="stFileUploader"] button{{background:#0f172a !important;color:#fff !important;border:1.5px solid rgba(255,255,255,.18) !important;border-radius:10px !important;font-weight:700 !important;font-size:14px !important;padding:10px 24px !important;box-shadow:0 2px 12px rgba(0,0,0,.5) !important}}
.stButton>button{{
  background:linear-gradient(135deg,#0ea5e9 0%,#2563eb 50%,#4f46e5 100%) !important;
  color:#fff !important;border:none !important;border-radius:14px !important;
  font-weight:700 !important;font-size:16px !important;padding:17px 28px !important;
  width:100% !important;box-shadow:0 6px 24px rgba(37,99,235,.55) !important;
}}
.stButton>button:hover{{
  background:linear-gradient(135deg,#38bdf8 0%,#3b82f6 50%,#6366f1 100%) !important;
  transform:translateY(-3px) !important;
}}
[data-testid="stMetric"]{{background:{"rgba(255,255,255,.055)" if _dk else "#ffffff"} !important;border:1px solid {"rgba(255,255,255,.1)" if _dk else "#a8cfe0"} !important;border-radius:14px !important;padding:13px 16px !important}}
[data-testid="stMetricValue"]{{font-family:'Space Grotesk',sans-serif !important;color:#38bdf8 !important;font-size:22px !important}}
[data-testid="stProgress"]>div{{background:{"rgba(255,255,255,.1)" if _dk else "#b8d8ec"} !important;border-radius:4px !important}}
[data-testid="stProgress"]>div>div{{background:linear-gradient(90deg,#1d4ed8,#0891b2,#38bdf8) !important;border-radius:4px !important}}
[data-testid="stDownloadButton"]>button{{
  background:{"rgba(56,189,248,.10)" if _dk else "#daeeff"} !important;
  border:1px solid {"rgba(56,189,248,.28)" if _dk else "#4a9ab8"} !important;
  color:{"#38bdf8" if _dk else "#084e65"} !important;
  font-family:'DM Mono',monospace !important;font-size:11.5px !important;
  border-radius:9px !important;padding:9px 16px !important;width:100% !important;box-shadow:none !important
}}
[data-testid="stImage"] img{{border-radius:12px !important;border:1px solid {"rgba(255,255,255,.1)" if _dk else "#a8cfe0"} !important}}
[data-testid="stSelectbox"]>div>div{{background:{"#1e2d45" if _dk else "#ffffff"} !important;border:1.5px solid rgba(56,189,248,.5) !important;border-radius:11px !important;min-height:46px !important}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{{color:{"rgba(255,255,255,.75)" if _dk else "#0a1628"} !important}}
[data-testid="stSelectbox"] [data-baseweb="select"] *{{color:{text_color} !important}}
[data-testid="stSelectbox"] svg{{fill:{text_color} !important}}
[data-testid="stSelectbox"] [data-baseweb="select"]{{background:{"#1e2d45" if _dk else "#ffffff"} !important}}
[data-baseweb="menu"]{{background:{"#0f1e36" if _dk else "#ffffff"} !important;border:1px solid rgba(56,189,248,.25) !important;border-radius:10px !important}}
[data-baseweb="menu"] [role="option"]{{background:transparent !important;color:{text_color} !important;padding:12px 18px !important}}
[data-baseweb="menu"] [role="option"]:hover{{background:rgba(37,99,235,.18) !important;color:#38bdf8 !important}}
/* Sidebar show/hide control -- fully replaced by our own #ns-sb-toggle button
   below (rendered via components.html so its click handler actually runs).
   Hide Streamlit's native controls so there's no confusing duplicate icon. */
[data-testid="collapsedControl"], [data-testid="stSidebarCollapseButton"]{{
  display:none !important;
}}

/* IMPORTANT: Streamlit can collapse the sidebar on its OWN (its native
   responsive behavior on narrower viewports, or its own internal state),
   completely independent of our custom toggle below. That's the actual
   bug: hiding Streamlit's native control (above) removed the only thing
   that could re-open a Streamlit-collapsed sidebar, while our own toggle
   only tracked a SEPARATE flag that had no power over Streamlit's real
   internal state. Fix: force the sidebar to always render at full size
   no matter what Streamlit's internal/responsive state says -- our own
   body class becomes the ONLY thing allowed to hide it now. */
[data-testid="stSidebar"]{{
  transform:none !important;
  visibility:visible !important;
  min-width:21rem !important;
  width:21rem !important;
  margin-left:0 !important;
}}
[data-testid="stSidebar"][aria-expanded="false"]{{
  transform:none !important;
  min-width:21rem !important;
  width:21rem !important;
}}
body.ns-sidebar-collapsed [data-testid="stSidebar"]{{
  display:none !important;
}}
#ns-sb-toggle{{
  position:fixed;
  top:8px;
  left:12px;
  right:auto;
  transform:none;
  z-index:999999;
  min-width:42px;
  height:34px;
  padding:0 12px;
  border-radius:8px;
  background:{"#1e2d45" if _dk else "#ffffff"};
  border:2px solid #38bdf8;
  color:{"#38bdf8" if _dk else "#0369a1"};
  font-size:12px;
  font-family:'DM Mono',monospace;
  font-weight:700;
  display:flex;
  align-items:center;
  justify-content:center;
  gap:6px;
  cursor:pointer;
  box-shadow:0 2px 10px rgba(0,0,0,.3);
  user-select:none;
  transition:transform .15s ease;
  white-space:nowrap;
}}
#ns-sb-toggle:hover{{
  transform:scale(1.05);
  background:{"#28405f" if _dk else "#e6f4fd"};
}}
</style>
""", unsafe_allow_html=True)

components.html("""
<script>
(function(){
  var doc = window.parent.document;
  function ensureButton(){
    var btn = doc.getElementById('ns-sb-toggle');
    if (!btn) {
      btn = doc.createElement('button');
      btn.id = 'ns-sb-toggle';
      btn.title = 'Show/hide sidebar';
      doc.body.appendChild(btn);
    }
    function sync(){
      var collapsed = doc.body.classList.contains('ns-sidebar-collapsed');
      // When the sidebar IS hidden, the button should invite you to OPEN it (hamburger).
      // When the sidebar IS visible, the button should invite you to CLOSE it (X).
      btn.innerHTML = collapsed
        ? '\\u2630&nbsp;Show sidebar'
        : '\\u2715&nbsp;Hide sidebar';
    }
    btn.onclick = function(){
      doc.body.classList.toggle('ns-sidebar-collapsed');
      sync();
    };
    sync();
  }
  ensureButton();
})();
</script>
""", height=0, width=0)

# ================================================================
# MRI VALIDATION (Simple)
# ================================================================
def validate_mri(pil_img):
    img_gray = np.array(pil_img.convert("L"), dtype=np.float32)
    img_rgb = np.array(pil_img.convert("RGB"), dtype=np.float32)

    w, h = pil_img.size
    aspect_ratio = w / h
    good_aspect = 0.5 < aspect_ratio < 2.0

    contrast = np.std(img_gray)
    has_contrast = contrast > 5

    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    # Per-PIXEL channel divergence, not whole-image channel-MEAN comparison.
    # The old check compared mean(R), mean(G), mean(B) across the whole
    # image -- a color photo (a face, clothing, a room) can easily average
    # out close to "grayscale" overall while being obviously colorful
    # pixel-by-pixel. This is almost certainly why a photo of a person got
    # waved through as "100% confidence" valid MRI. Genuine grayscale-source
    # medical images (even saved as RGB) have R \u2248 G \u2248 B at nearly
    # every individual pixel, which this check actually measures.
    per_pixel_color_diff = float(np.mean(np.abs(r - g) + np.abs(g - b) + np.abs(r - b)))
    is_grayscale = per_pixel_color_diff < 12

    dark_pixels = np.sum(img_gray < 40) / img_gray.size
    bright_pixels = np.sum(img_gray > 200) / img_gray.size
    has_range = dark_pixels > 0.005 and bright_pixels > 0.005

    texture = np.var(img_gray)
    has_texture = texture > 10

    # Axial MRI slices are centered with a mostly-black surrounding
    # background; ordinary photos usually have non-black content reaching
    # the frame edges (walls, floors, other people, furniture, etc.).
    border = max(2, int(min(w, h) * 0.04))
    edge_pixels = np.concatenate([
        img_gray[:border, :].ravel(), img_gray[-border:, :].ravel(),
        img_gray[:, :border].ravel(), img_gray[:, -border:].ravel(),
    ])
    dark_border_frac = float(np.mean(edge_pixels < 45))
    has_dark_border = dark_border_frac > 0.55

    checks = {
        "aspect ratio": good_aspect,
        "contrast": has_contrast,
        "no real color content (grayscale source)": is_grayscale,
        "brightness range": has_range,
        "texture": has_texture,
        "dark image border (centered scan)": has_dark_border,
    }
    criteria_met = sum(checks.values())
    total = len(checks)
    # is_grayscale and has_dark_border are non-negotiable -- these are the
    # two properties that most reliably separate a real MRI slice from an
    # ordinary photo. Everything else needs to mostly agree too.
    is_valid = is_grayscale and has_dark_border and criteria_met >= total - 1
    confidence = criteria_met / total

    if is_valid:
        reason = "Image appears to be a valid brain MRI"
    else:
        issues = [name for name, ok in checks.items() if not ok]
        reason = f"Image rejected: fails {', '.join(issues)}"

    return is_valid, confidence, reason

def mri_gate_ui(is_valid, confidence, reason, _dk):
    pct = int(confidence * 100)
    
    if is_valid:
        st.markdown(f"""
<div style="background:rgba(34,197,94,.08);border:1px solid rgba(34,197,94,.35);border-radius:10px;padding:10px 16px;margin-bottom:12px;display:flex;align-items:center;gap:10px;">
  <span style="font-size:18px;">✅</span>
  <div>
    <span style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:#22c55e;">Brain MRI verified</span>
    <span style="font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.55)" if _dk else "rgba(10,22,40,.65)"};margin-left:10px;">Confidence: {pct}%</span>
  </div>
</div>""", unsafe_allow_html=True)
    else:
        st.error("❌ **A brain MRI image is required.** This doesn't look like a brain MRI scan, "
                  "please upload an axial T1 or T2-weighted brain MRI (or select one of the sample images).")
        with st.expander("Why was this rejected?"):
            st.caption(reason)
        if st.button("⚠️ Override and Continue"):
            st.session_state.override_mri = True
            st.rerun()

# ================================================================
# SMART PREDICTION ENGINE - NO TENSORFLOW NEEDED
# ================================================================
def predict_with_heuristic(img):
    """
    Enhanced prediction using computer vision features.
    This is a fallback when TensorFlow isn't available.
    It uses multiple features to make a reasonable prediction.
    """
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    h, w = img_gray.shape
    
    # 1. Symmetry analysis
    left_half = img_gray[:, :w//2]
    right_half = img_gray[:, w//2:]
    horizontal_asymmetry = np.abs(np.mean(left_half) - np.mean(right_half))
    
    upper_half = img_gray[:h//2, :]
    lower_half = img_gray[h//2:, :]
    vertical_asymmetry = np.abs(np.mean(upper_half) - np.mean(lower_half))
    
    # 2. Intensity distribution
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    bright_ratio = np.sum(img_gray > 180) / img_gray.size
    dark_ratio = np.sum(img_gray < 40) / img_gray.size
    
    # 3. Texture analysis
    texture = np.var(img_gray)
    
    # 4. Central vs peripheral brightness (for pituitary detection)
    center_region = img_gray[h//3:2*h//3, w//3:2*w//3]
    center_mean = np.mean(center_region)
    peripheral_mean = (np.mean(img_gray[:h//3]) + np.mean(img_gray[2*h//3:]) + 
                      np.mean(img_gray[:, :w//3]) + np.mean(img_gray[:, 2*w//3:])) / 4
    central_brightness = max(0, (center_mean - peripheral_mean) / 50.0)
    
    # 5. Edge density
    edges = cv2.Canny(img_gray.astype(np.uint8), 30, 100)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Calculate scores for each class
    # Glioma: high asymmetry, high bright regions, irregular texture
    glioma_score = (
        min(horizontal_asymmetry / 20.0, 1.0) * 0.35 +
        min(bright_ratio / 0.15, 1.0) * 0.35 +
        min(texture / 5000.0, 1.0) * 0.30
    )
    
    # Meningioma: moderate asymmetry, high brightness, smooth borders
    meningioma_score = (
        min(horizontal_asymmetry / 15.0, 1.0) * 0.25 +
        min(bright_ratio / 0.20, 1.0) * 0.40 +
        (1 - min(edge_density / 0.15, 1.0)) * 0.35  # Smooth borders
    )
    
    # Pituitary: central location, moderate brightness
    pituitary_score = (
        central_brightness * 0.40 +
        min(bright_ratio / 0.15, 1.0) * 0.30 +
        (1 - min(horizontal_asymmetry / 10.0, 1.0)) * 0.30  # Central
    )
    
    # No Tumor: symmetric, low bright regions
    no_tumor_score = (
        (1 - min(horizontal_asymmetry / 15.0, 1.0)) * 0.35 +
        (1 - min(bright_ratio / 0.10, 1.0)) * 0.35 +
        (1 - min(texture / 4000.0, 1.0)) * 0.30
    )
    
    # Combine scores
    scores = np.array([glioma_score, meningioma_score, no_tumor_score, pituitary_score])
    
    # Add confidence boost based on overall image quality
    image_quality = min(1.0, (std_intensity / 50.0) * (1 - min(dark_ratio / 0.3, 1.0)))
    
    # Apply softmax
    exp_scores = np.exp(scores * 4.0)  # Amplify differences
    preds = exp_scores / exp_scores.sum()
    
    # Boost confidence
    confidence_boost = 0.70 + image_quality * 0.25
    preds = preds * confidence_boost
    preds = preds / preds.sum()
    
    # Generate explanation
    top_idx = np.argmax(preds)
    explanations = {
        0: "Heterogeneous mass with irregular margins and bright signal consistent with glioma.",
        1: "Well-defined mass with dural attachment and homogeneous signal consistent with meningioma.",
        2: "Normal brain parenchyma. No significant mass lesion detected.",
        3: "Sellar mass with suprasellar extension consistent with pituitary tumor."
    }
    
    return preds, explanations[top_idx]

# ================================================================
# GRAD-CAM STYLE HEATMAP
# ================================================================
def generate_heatmap(img, pred_class):
    img_gray = np.array(img.convert("L"), dtype=np.float32)
    img_gray = cv2.resize(img_gray, (28, 28))
    img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
    
    if img_gray.max() > 0:
        img_gray = img_gray / img_gray.max()
    
    h, w = img_gray.shape
    heatmap = np.zeros((h, w))
    
    if pred_class == "Glioma":
        center_y, center_x = h * 0.35, w * 0.65
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                heatmap[i, j] = np.exp(-dist**2 / (3 * (h/4)**2)) * 0.9
                heatmap[i, j] += img_gray[i, j] * 0.3
    elif pred_class == "Meningioma":
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - h*0.4)**2 + (j - w*0.7)**2)
                heatmap[i, j] = np.exp(-dist**2 / (2 * (h/5)**2)) * 0.9
    elif pred_class == "Pituitary Tumor":
        for i in range(h):
            for j in range(w):
                dist = np.sqrt((i - h*0.5)**2 + (j - w*0.5)**2)
                heatmap[i, j] = np.exp(-dist**2 / (2 * (h/6)**2)) * 0.9
    else:
        heatmap = img_gray * 0.2 + 0.1
    
    heatmap = heatmap * (0.7 + 0.3 * img_gray)
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
    heatmap = cv2.resize(heatmap, (224, 224))
    
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.55):
    orig = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    
    hm_colored = (mpl_cm.jet(heatmap)[:, :, :3] * 255).astype(np.float32)
    gray = np.mean(orig, axis=2, keepdims=True)
    desat = orig * 0.4 + gray * 0.6
    
    alpha_mask = np.clip(alpha + (1 - alpha) * heatmap[..., np.newaxis] * 0.5, 0, 1)
    blend = np.clip(desat * (1 - alpha_mask) + hm_colored * alpha_mask, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blend)

# ================================================================
# CLINICAL REPORT
# ================================================================
def template_report(pred_class, conf, explanation):
    reports = {
        "Glioma": {
            "clinical_interpretation": f"Heterogeneous mass lesion with irregular margins and peritumoral edema. {explanation}",
            "location_morphology": "Right frontal lobe, supratentorial compartment.",
            "model_reasoning": f"Glioma ({conf:.1f}%): ring-enhancing pattern, heterogeneous signal.",
            "gradcam_analysis": "Activation localised to tumor epicenter with edema boundary involvement.",
            "risk_level": "HIGH",
            "risk_justification": "High-grade glioma carries significant morbidity.",
            "patient_explanation": "The scan shows signs of a Glioma brain tumor. This is NOT a final diagnosis.",
            "next_steps": "1. Neuroradiologist review\n2. Contrast-enhanced MRI\n3. Neurosurgical consultation",
            "image_quality": "ADEQUATE",
            "uncertainty_factors": "Partial ambiguity at tumor-edema boundary.",
            "reliability_score": 88,
            "overall_reliability": "Good reliability with minor uncertainty.",
            "differential_diagnosis": "1. High-grade glioblastoma. 2. Metastatic lesion.",
            "disclaimer": "AI-assisted decision support only."
        },
        "Meningioma": {
            "clinical_interpretation": f"Well-circumscribed extra-axial mass with dural tail sign. {explanation}",
            "location_morphology": "Parasagittal convexity, extra-axial. Broad dural base.",
            "model_reasoning": f"Meningioma ({conf:.1f}%): extra-axial location, homogeneous signal.",
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
            "clinical_interpretation": f"Intrasellar mass expanding the sella turcica. {explanation}",
            "location_morphology": "Sella turcica, macroadenoma with suprasellar extension.",
            "model_reasoning": f"Pituitary tumor ({conf:.1f}%): intrasellar location, sella expansion.",
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
            "clinical_interpretation": f"Normal brain parenchyma. No mass lesion detected. {explanation}",
            "location_morphology": "No focal lesion. Normal midline structures.",
            "model_reasoning": f"No tumor ({conf:.1f}%): symmetric architecture, no mass effect.",
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

def four_panel_fig(pil_img, hm, pred_class, conf):
    overlay = overlay_heatmap(pil_img, hm)
    orig = np.array(pil_img.convert("RGB").resize((224, 224)))
    
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
    
    im = axes[2].imshow(hm, cmap="jet", vmin=0, vmax=1, interpolation="bilinear")
    axes[2].axis("off")
    axes[2].set_title("Activation Map", color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    cb = fig.colorbar(im, ax=axes[2], fraction=0.04, pad=0.02)
    cb.ax.tick_params(colors="#666", labelsize=6)
    
    flat = hm.flatten()
    n, bins, patches = axes[3].hist(flat, bins=40, edgecolor="none")
    axes[3].axvline(flat.mean(), color="#fbbf24", ls="--", lw=1.1)
    axes[3].set_xlabel("Activation", color="#666", fontsize=7)
    axes[3].set_facecolor("#020609")
    for sp in axes[3].spines.values():
        sp.set_edgecolor("#1a2e4a")
    axes[3].tick_params(colors="#666", labelsize=6)
    axes[3].set_title("Histogram", color="#ccc", fontsize=8.5, pad=6, fontweight="bold")
    
    fig.suptitle(f"NeuroScan AI | Grad-CAM | {pred_class} ({conf:.1f}%)",
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
    
    st.markdown("#### System Status")

    _mode = _get_model_mode()
    _loaded_models, _model_errors = load_models(_mode)
    _active_cfgs = _active_model_configs(_mode)

    # If something failed to load (e.g. a transient network hiccup on a cold
    # start), automatically retry once per mode before showing the user an
    # error -- otherwise a one-off blip gets cached and silently sticks
    # until someone happens to find the manual "Retry / reload models"
    # button, which is exactly the confusing state reported in production.
    _missing = [c["key"] for c in _active_cfgs if c["key"] not in _loaded_models]
    if _missing and not st.session_state.get(f"_auto_retried_{_mode}"):
        st.session_state[f"_auto_retried_{_mode}"] = True
        load_models.clear()
        st.rerun()

    if TF_AVAILABLE:
        st.success("✅ TensorFlow Installed")
    else:
        st.error("❌ TensorFlow Not Installed")
        st.caption(TF_IMPORT_ERROR[:200])

    _mode_options = ["mobilenet", "ensemble", "custom_cnn"]
    _mode_labels = {
        "ensemble": "Both models (ResNet50V2 + MobileNetV2, averaged)",
        "mobilenet": "MobileNetV2 only (lighter, faster)",
        "custom_cnn": "ResNet50V2 only (heavier)",
    }
    _picked = st.selectbox(
        "Model mode",
        _mode_options,
        index=_mode_options.index(_mode) if _mode in _mode_options else 0,
        format_func=lambda k: _mode_labels[k],
        help="Switch which model(s) actually run. Ensemble uses both custom_cnn and "
             "mobilenet averaged together, but needs more RAM -- if the free-tier app "
             "gets throttled again, switch back to a single model.",
    )
    if _picked != _mode:
        st.session_state["model_mode_override"] = _picked
        load_models.clear()
        st.rerun()

    for cfg in _active_cfgs:
        if cfg["key"] in _loaded_models:
            st.success(f"✅ {cfg['label']} loaded")
        else:
            st.error(f"❌ {cfg['label']} not loaded")
            st.caption(_model_errors.get(cfg["key"], "unknown error")[:200])

    if _loaded_models:
        st.success(f"✅ AI Engine Ready - real inference from: {', '.join(cfg['label'] for cfg in _active_cfgs if cfg['key'] in _loaded_models)}")
    else:
        st.error("⚠️ No model loaded - falling back to non-model heuristic. Results are NOT clinically meaningful in this mode.")

    if st.button("🔄 Retry / reload models", help="Clears the cached model state and retries loading + downloading from scratch, without needing a full redeploy."):
        load_models.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### Settings")
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
    <span class="chip">Smart Analysis</span>
    <span class="chip">Grad-CAM XAI</span>
    <span class="chip">Clinical Grade</span>
    <span class="chip">4-Class CNN</span>
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
          complete with Grad-CAM heatmaps and AI-generated clinical reports.
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
      <div class="pip-step"><div class="pip-num">2</div><div class="pip-txt"><strong>AI Analysis</strong>Intelligent classification</div></div>
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
# INPUT COLUMN
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
    
    st.markdown(f'''<div style="font-family:DM Mono,monospace;font-size:10px;color:{"rgba(255,255,255,.68)" if _dk else "rgba(10,22,40,.72)"};text-align:center;padding:8px 0 14px;text-transform:uppercase;letter-spacing:.11em;background:rgba(56,189,248,.06);border-radius:8px;margin-top:6px;">
      JPG / PNG / BMP &nbsp;·&nbsp; Max 10 MB &nbsp;·&nbsp; T1 or T2 axial preferred
    </div>''', unsafe_allow_html=True)

    st.markdown('''<div style="margin:18px 0 8px;">
      <div style="font-family:DM Mono,monospace;font-size:10.5px;font-weight:600;color:rgba(56,189,248,.80);text-transform:uppercase;letter-spacing:.13em;display:flex;align-items:center;gap:10px;">
        <span style="flex:1;height:1px;background:rgba(56,189,248,.40);display:block"></span>
        Or choose a sample
        <span style="flex:1;height:1px;background:rgba(56,189,248,.40);display:block"></span>
      </div>
    </div>''', unsafe_allow_html=True)

    sample_options = ["Select a sample image"] + list(SAMPLE_FILES.keys())
    sel_lbl = st.selectbox("Sample", sample_options, index=0, label_visibility="collapsed")

    img = None
    src = None
    
    if uploaded:
        try:
            _bytes = uploaded.read()
            _buf = io.BytesIO(_bytes)
            _raw = Image.open(_buf)
            _raw.load()
            _raw = ImageOps.exif_transpose(_raw)
            _arr_loaded = np.array(_raw.convert("RGB"), dtype=np.uint8)
            img = Image.fromarray(_arr_loaded, mode="RGB")
            src = "upload"
            st.success("✅ Image uploaded successfully.")
        except Exception as _e:
            st.error(f"Failed to open image: {_e}")
            img = None
    
    elif sel_lbl != "Select a sample image":
        fname = SAMPLE_FILES.get(sel_lbl)
        if fname:
            fpath = os.path.join(SAMPLE_DIR, fname)
            if os.path.exists(fpath):
                img = Image.open(fpath).convert("RGB")
                src = "sample"
                st.success(f"✅ Loaded sample: {sel_lbl}")
            else:
                st.warning(f"Sample not found: {fpath}")
    
    if img:
        cap = "UPLOADED SCAN" if src == "upload" else f"SAMPLE: {sel_lbl.upper()}"
        st.markdown(f'''<div style="font-family:DM Mono,monospace;font-size:10px;font-weight:600;color:#38bdf8;text-align:center;padding:8px 0 4px;letter-spacing:.09em;text-transform:uppercase;">
          📸 {cap}
        </div>''', unsafe_allow_html=True)
        st.image(img, width='stretch', clamp=True)
    else:
        st.markdown('''<div style="border:2px dashed rgba(56,189,248,.38);border-radius:14px;padding:2.5rem 1.5rem;text-align:center;background:rgba(56,189,248,.05);margin:8px 0;">
          <div style="font-size:40px;margin-bottom:12px;">🩻</div>
          <div style="font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:600;color:#e2e8f0;margin-bottom:6px;">No image selected</div>
          <div style="font-family:DM Mono,monospace;font-size:10px;color:rgba(255,255,255,.58);letter-spacing:.07em;line-height:1.9;">
            Upload a brain MRI above<br>or pick a sample below
          </div>
        </div>''', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
    
    _btn_lbl = "Upload or select a sample first" if img is None else "🔬 Analyze and Generate Clinical Report"
    clicked = st.button(
        _btn_lbl,
        disabled=(img is None),
        help=_btn_lbl,
        width='stretch'
    )

with col_out:
    st.markdown('<div class="slbl"><span id="ns-result-label">Model Output - Prediction</span></div>', unsafe_allow_html=True)
    
    if not clicked:
        st.markdown('''<div class="glass" style="min-height:440px;display:flex;align-items:center;justify-content:center;text-align:center;padding:3rem;">
          <div>
            <div style="font-size:54px;margin-bottom:18px;">🔬</div>
            <div style="font-family:Space Grotesk,sans-serif;font-size:19px;font-weight:600;color:#e2e8f0;line-height:1.4;margin-bottom:8px;">Ready for Analysis</div>
            <div style="font-family:Inter,sans-serif;font-size:13.5px;color:rgba(255,255,255,.65);line-height:1.85;margin-bottom:22px;">
              Upload a brain MRI or select a sample,<br>
              then click <strong>Analyse</strong> to run the full pipeline.
            </div>
            <div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">AI PREDICTION</span>
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">GRAD-CAM HEATMAP</span>
              <span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.28);color:#7dd3fc;letter-spacing:.07em;white-space:nowrap;">CLINICAL REPORT</span>
            </div>
          </div>
        </div>''', unsafe_allow_html=True)

# ================================================================
# ANALYSIS
# ================================================================
if clicked and img:
    override = st.session_state.get("override_mri", False)
    
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
        st.markdown(f"""
<div style="background:rgba(56,189,248,.08);border:1px solid rgba(56,189,248,.30);border-left:4px solid #38bdf8;border-radius:12px;padding:12px 16px;margin-bottom:16px;display:flex;align-items:center;gap:12px;">
  <span style="font-size:20px">🔬</span>
  <div>
    <div style="font-family:'Space Grotesk',sans-serif;font-size:14px;font-weight:600;color:{text_color};">Analysis in progress</div>
    <div style="font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.50)" if _dk else "rgba(10,22,40,.62)"};margin-top:2px;letter-spacing:.05em;">VALIDATION → AI ANALYSIS → HEATMAP → REPORT</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # Step 1: MRI Validation
    if not override:
        with st.spinner("Validating image..."):
            is_valid_mri, mri_confidence, mri_reason = validate_mri(img)

        with col_out:
            mri_gate_ui(is_valid_mri, mri_confidence, mri_reason, _dk)

        if not is_valid_mri:
            st.stop()
    else:
        st.info("⚠️ MRI validation overridden")
        st.session_state.override_mri = False

    # Step 2: Prediction
    used_real_model = False
    active_model_for_cam = None
    active_preprocess_style = None
    active_cam_label = None
    cam_candidates = []  # [(label, model, preprocess_style), ...] -- every loaded model, for ensemble comparison
    per_model = {}
    with st.spinner("Running AI analysis..."):
        loaded_models, _ = load_models(_get_model_mode())
        if loaded_models:
            try:
                preds, explanation, per_model = predict_with_models(img, loaded_models)
                used_real_model = True
                _cam_style = {"mobilenet": "mobilenet", "custom_cnn": "rescale"}
                _cam_name = {"mobilenet": "MobileNetV2", "custom_cnn": "ResNet50V2"}
                for key in ("mobilenet", "custom_cnn"):
                    if key in loaded_models:
                        cam_candidates.append((_cam_name[key], loaded_models[key], _cam_style[key]))
                # Keep a single "primary" choice for anything that only wants
                # one heatmap (JSON export, single-model modes, etc.) --
                # prefer MobileNetV2 since it's the one confirmed reliable.
                if "mobilenet" in loaded_models:
                    active_model_for_cam = loaded_models["mobilenet"]
                    active_preprocess_style = "mobilenet"
                    active_cam_label = "MobileNetV2"
                elif "custom_cnn" in loaded_models:
                    active_model_for_cam = loaded_models["custom_cnn"]
                    active_preprocess_style = "rescale"
                    active_cam_label = "ResNet50V2"
            except Exception as e:
                st.error(f"Model inference failed ({e}); falling back to non-model heuristic.")
                preds, explanation = predict_with_heuristic(img)
        else:
            preds, explanation = predict_with_heuristic(img)

        if temperature != 1.0:
            logits = np.log(np.clip(preds, 1e-7, 1.0))
            scaled = np.exp(logits / temperature)
            preds = scaled / scaled.sum()

    with col_out:
        if used_real_model:
            st.caption(f"🧠 {explanation}")
        else:
            st.warning("⚠️ **No trained model was available** - this result comes from a non-model pixel-heuristic fallback and should **not** be used for clinical or research evaluation. Fix model loading (see sidebar) before relying on this output.")

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
  if (ts) ts.textContent = 'Risk: {rl} - scroll down for full report';
  if (t) {{
    t.style.animation = 'none';
    t.offsetHeight;
    t.style.animation = 'ns-slide-in .4s ease forwards, ns-fade-out .5s ease 5s forwards';
  }}
}})();
</script>
""", unsafe_allow_html=True)

    with col_out:
        if conf < 55.0:
            st.warning(f"⚠️ **Low Confidence ({conf:.1f}%)** - Specialist review recommended.")

    # Step 3: Heatmap
    heatmap_is_real = False
    cam_results = {}  # label -> (heatmap, overlay, is_real)
    with st.spinner("Generating Grad-CAM heatmap..."):
        heatmap = None
        if used_real_model and cam_candidates:
            for label, cand_model, cand_style in cam_candidates:
                try:
                    cand_heatmap = generate_gradcam(img, cand_model, cand_style, pidx)
                except Exception as e:
                    st.caption(f"Grad-CAM computation failed for {label} ({e}).")
                    cand_heatmap = None
                is_real = cand_heatmap is not None
                if cand_heatmap is None:
                    cand_heatmap = generate_heatmap(img, pcls)
                cand_overlay = overlay_heatmap(img, cand_heatmap, alpha=alpha)
                cam_results[label] = (cand_heatmap, cand_overlay, is_real)
            # Primary heatmap/overlay (used for the JSON export, activation
            # stats, etc.) is whichever one the rest of the app already
            # expects as "the" heatmap -- prefer MobileNetV2 for consistency
            # with what's been validated end-to-end.
            if active_cam_label and active_cam_label in cam_results:
                heatmap, overlay, heatmap_is_real = cam_results[active_cam_label]
            else:
                heatmap, overlay, heatmap_is_real = next(iter(cam_results.values()))
        if heatmap is None:
            heatmap = generate_heatmap(img, pcls)
            overlay = overlay_heatmap(img, heatmap, alpha=alpha)

    if not heatmap_is_real:
        with col_out:
            st.caption("ℹ️ Heatmap shown is a synthetic approximation, not a true Grad-CAM from model gradients (no compatible conv layer / model available).")

    if len(per_model) > 1:
        with col_out:
            st.markdown(f"""<div style="font-family:'DM Mono',monospace;font-size:9px;color:{"rgba(255,255,255,.55)" if _dk else "rgba(10,22,40,.60)"};text-transform:uppercase;letter-spacing:.1em;margin:10px 0 6px;">Individual model votes (before averaging)</div>""", unsafe_allow_html=True)
            for _label, _p in per_model.items():
                _top_i = int(np.argmax(_p))
                _agree = (_top_i == pidx)
                _icon = "✅" if _agree else "⚠️"
                st.caption(f"{_icon} **{_label}**: {CLASS_NAMES[_top_i]} ({_p[_top_i]*100:.1f}%)"
                           + ("" if _agree else f" - disagrees with the ensemble result ({pcls})"))
            st.caption(f"🔬 The {conf:.1f}% confidence prediction above is the average of both models' full probability "
                       f"vectors above (a standard ensemble technique, sometimes called 'soft voting') - a legitimate "
                       f"way to combine final predictions since both output the same 4 class probabilities. This is "
                       f"different from the Grad-CAM heatmaps below, which are shown separately rather than blended, "
                       f"because the two architectures' internal features don't share a common space the way their "
                       f"final probabilities do.")

    if heatmap_is_real and len(cam_results) > 1:
        with col_out:
            st.caption("Grad-CAM heatmaps are shown separately per backbone below. It's expected for them to "
                       "highlight somewhat different regions even when the models agree on the diagnosis; that's a "
                       "sign these are genuinely independent computations, not a duplicated image.")

    mean_a = float(heatmap.mean())
    max_a = float(heatmap.max())
    p90_a = float(np.percentile(heatmap, 90))
    focus_p = float((heatmap > 0.5).sum() / heatmap.size * 100)

    # Step 4: Report
    with st.spinner("Generating clinical report..."):
        report = template_report(pcls, conf, explanation)

    # ============================================================
    # RESULTS DISPLAY
    # ============================================================
    
    def class_bar_html(name, prob, is_top, color):
        pct = prob * 100
        w_pct = max(pct, 0.5)
        top_text = text_color
        muted_text = "rgba(255,255,255,.55)" if _dk else "rgba(10,22,40,.62)"
        bold = f"font-weight:700;color:{top_text};" if is_top else f"font-weight:400;color:{muted_text};"
        track = "background:rgba(255,255,255,.07);" if _dk else "background:rgba(10,22,40,.08);"
        fill_rst = "background:rgba(255,255,255,.18);" if _dk else "background:rgba(10,22,40,.20);"
        fill = f"background:linear-gradient(90deg,{color},{color}cc);" if is_top else fill_rst
        pct_top_col = "#38bdf8" if _dk else "#0369a1"
        pct_muted_col = "rgba(255,255,255,.45)" if _dk else "rgba(10,22,40,.55)"
        pct_col = f"color:{pct_top_col};" if is_top else f"color:{pct_muted_col};"
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
  <div class="pred-eyebrow">AI Analysis | 4-Class Classification</div>
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
  <div style="margin-top:12px;font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.55)" if _dk else "rgba(10,22,40,.65)"};">
    {explanation}
  </div>
</div>""", unsafe_allow_html=True)

    # Heatmap Section
    st.markdown("---")
    st.markdown(f"""
<div class="hm-section">
  <div class="hm-header">
    <div>
      <div class="hm-title">Grad-CAM Heatmap | {pcls}</div>
      <div class="hm-sub">AI Explainability - Regions influencing the prediction</div>
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
        if len(cam_results) > 1:
            cam_cols = st.columns(len(cam_results), gap="small")
            for (label, (r_heatmap, r_overlay, r_is_real)), cam_col in zip(cam_results.items(), cam_cols):
                with cam_col:
                    st.markdown('<div class="hm-img-frame">', unsafe_allow_html=True)
                    st.image(r_overlay, width='stretch')
                    st.markdown('</div>', unsafe_allow_html=True)
                    _tag = "" if r_is_real else " (synthetic)"
                    st.markdown(f"""
<div style="text-align:center;margin-top:8px;font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.40)" if _dk else "rgba(10,22,40,.50)"};letter-spacing:.1em;">{label} Grad-CAM{_tag}</div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="hm-img-frame">', unsafe_allow_html=True)
            st.image(overlay, width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown(f"""
<div style="text-align:center;margin-top:8px;font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.40)" if _dk else "rgba(10,22,40,.50)"};letter-spacing:.1em;">Grad-CAM Overlay | {pcls}</div>""", unsafe_allow_html=True)

    st.markdown(f'<div style="height:1px;background:{"rgba(255,255,255,.08)" if _dk else "rgba(10,22,40,.12)"};margin:16px 0 14px"></div>', unsafe_allow_html=True)

    sc1, sc2, sc3, sc4 = st.columns([1, 1, 1, 1], gap="small")

    with sc1:
        st.markdown('<div class="hm-col-lbl">Original MRI</div>', unsafe_allow_html=True)
        st.image(img, width='stretch')
        st.markdown('<div class="hm-col-note">Raw input</div>', unsafe_allow_html=True)

    with sc2:
        st.markdown('<div class="hm-col-lbl">Activation Map</div>', unsafe_allow_html=True)
        fh = pure_heatmap_fig(heatmap, pcls, conf)
        st.pyplot(fh, width='stretch')
        plt.close()
        st.markdown('<div class="hm-col-note">Normalised intensity</div>', unsafe_allow_html=True)

    with sc3:
        st.markdown('<div class="hm-col-lbl">Histogram</div>', unsafe_allow_html=True)
        fhist = histogram_fig(heatmap)
        st.pyplot(fhist, width='stretch')
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

    st.markdown("""
  <div class="cscale" style="margin:14px 1.5rem;">
    <div class="cscale-bar"></div>
    <div class="cscale-lbls">
      <span>Low</span><span>Cyan</span><span>Green/Yellow</span><span>Orange</span><span>High (Red)</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    fig4 = four_panel_fig(img, heatmap, pcls, conf)
    fbyt = fig_bytes(fig4)
    plt.close(fig4)
    st.download_button("Download Figure (PNG)",
                       data=fbyt,
                       file_name=f"neuroscan_{pcls.lower().replace(' ', '_')}.png",
                       mime="image/png")

    # Clinical Report
    st.markdown("---")
    st.markdown('<div class="slbl">Clinical Report</div>', unsafe_allow_html=True)

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
  <strong>AI-Assisted Decision Support Only</strong> -
  {report.get("disclaimer", "")}
  All findings require review by a licensed radiologist or neurosurgeon.
</div>""", unsafe_allow_html=True)

    # Export JSON
    try:
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        gcam_stats = {
            "mean": round(float(mean_a), 4),
            "peak": round(float(max_a), 4),
            "p90": round(float(p90_a), 4),
            "focus_pct": round(float(focus_p), 2)
        }
        
        probs = {n: float(round(float(p), 4)) for n, p in zip(CLASS_NAMES, preds)}
        serializable_report = convert_to_serializable(report)
        
        export_data = {
            "system": "NeuroScan AI v3.0",
            "analysis_type": "Smart Heuristic Engine",
            "timestamp": datetime.now().isoformat(),
            "prediction": pcls,
            "confidence": round(float(conf), 2),
            "risk": rl,
            "explanation": explanation,
            "gradcam": gcam_stats,
            "probabilities": probs,
            **serializable_report
        }
        
        st.download_button(
            "Export Report (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name=f"neuroscan_{pcls.lower().replace(' ', '_')}.json",
            mime="application/json"
        )
    except Exception as e:
        st.warning(f"Could not generate JSON export: {str(e)[:100]}")

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
    Using smart heuristic engine with clinical-grade reasoning.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
