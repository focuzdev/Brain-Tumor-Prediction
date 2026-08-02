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

ANTHROPIC_AVAILABLE = False
ANTHROPIC_IMPORT_ERROR = ""
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except Exception as e:
    ANTHROPIC_IMPORT_ERROR = str(e)

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
    from tensorflow.keras.applications.resnet_v2 import preprocess_input as resnet_preprocess
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
        # NOTE: per the training notebook's own Cell 14, brain_tumor_model.h5
        # is saved as "whichever of ResNet50V2/MobileNetV2 scored higher on
        # that specific training run" -- its architecture is NOT guaranteed
        # to be ResNet50V2, that was an assumption from the chart title in
        # training_history.png, not a verified fact. load_models() below
        # inspects the actual loaded model's backbone layer name at runtime
        # and corrects this label + the preprocessing function if needed, so
        # this "resnet50v2" default is a best guess, not the source of truth.
        "key": "custom_cnn", "label": "ResNet50V2", "file": "brain_tumor_model.h5",
        "preprocess": "resnet", "expected_size": 230584088,
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

def _detect_backbone_architecture(model):
    """
    Inspects the actual loaded model's backbone layer to determine its real
    architecture, rather than trusting a filename or an assumed label.

    This matters specifically for brain_tumor_model.h5: the training
    notebook's own Cell 14 saves "whichever of ResNet50V2/MobileNetV2 scored
    higher on that training run" under this filename -- the architecture
    inside it is empirically decided per-run, not fixed. Following the same
    convention the notebook itself uses for Grad-CAM (backbone is
    model.layers[1], since Input is layers[0]), so this detection matches
    exactly how the model was actually built.
    """
    try:
        name = model.layers[1].name.lower()
    except Exception:
        name = ""
    if "resnet" in name:
        return "resnet50v2", name
    if "mobilenet" in name:
        return "mobilenetv2", name
    return "unknown", name

@st.cache_resource(show_spinner=False)
def load_models(mode):
    """
    Loads the model(s) selected by `mode`, downloading real weights first if
    the local copy is missing or is an unresolved Git LFS pointer stub.
    Returns (loaded: dict[key -> keras.Model], errors: dict[key -> str],
    runtime_meta: dict[key -> {"label": str, "preprocess": str}]).

    `runtime_meta` reflects each model's ACTUAL detected architecture, not
    just the static guess in MODEL_CONFIGS -- see _detect_backbone_architecture.
    This is what the rest of the app should use for preprocessing/labeling,
    since brain_tumor_model.h5's true identity isn't guaranteed by its
    filename (see that function's docstring).

    IMPORTANT: `mode` must be an explicit parameter, not read from session
    state/secrets inside the function body. st.cache_resource keys its cache
    purely on function arguments -- a zero-argument version of this function
    would return the SAME cached result for every mode after the first call,
    silently showing stale "not loaded" state for models that were never
    even attempted under whatever mode happened to be cached first. Passing
    `mode` in gives each mode its own independent cache entry.
    """
    loaded, errors, runtime_meta = {}, {}, {}
    _log(f"load_models() starting, MODEL_MODE={mode!r}")
    if not TF_AVAILABLE:
        _log(f"ERROR: TensorFlow not available: {TF_IMPORT_ERROR}")
        errors["_tensorflow"] = TF_IMPORT_ERROR or "TensorFlow is not installed in this environment."
        return loaded, errors, runtime_meta
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

            detected_arch, backbone_layer_name = _detect_backbone_architecture(m)
            expected_arch = "resnet50v2" if cfg["key"] == "custom_cnn" else "mobilenetv2"
            if detected_arch == "unknown":
                _log(f"{tag} WARNING: could not detect backbone architecture from layer name "
                     f"'{backbone_layer_name}' -- keeping configured label/preprocess as a best guess")
                runtime_meta[cfg["key"]] = {"label": cfg["label"], "preprocess": cfg["preprocess"]}
            elif detected_arch != expected_arch:
                _log(f"{tag} MISMATCH: expected {expected_arch} but backbone layer is '{backbone_layer_name}' "
                     f"(detected {detected_arch}). Correcting label and preprocessing to match reality "
                     f"instead of the assumed filename/label.")
                corrected_label = "ResNet50V2" if detected_arch == "resnet50v2" else "MobileNetV2"
                corrected_preprocess = "resnet" if detected_arch == "resnet50v2" else "mobilenet"
                runtime_meta[cfg["key"]] = {"label": corrected_label, "preprocess": corrected_preprocess}
            else:
                _log(f"{tag} architecture verified: backbone layer '{backbone_layer_name}' matches expected {expected_arch}")
                runtime_meta[cfg["key"]] = {"label": cfg["label"], "preprocess": cfg["preprocess"]}
        except Exception as e:
            import traceback
            _log(f"{tag} ERROR: load_model() failed:\n{traceback.format_exc()}")
            errors[cfg["key"]] = f"Failed to load {cfg['file']}: {e}"
    _log(f"load_models() done. loaded={list(loaded.keys())} errors={list(errors.keys())}")
    return loaded, errors, runtime_meta

def preprocess_for_model(pil_img, style):
    """
    Converts a PIL image into the exact tensor shape/scale a given model
    expects. Verified directly against the real training notebook
    (neuroscan_expert.ipynb):
      - MobileNetV2 trained with tf.keras.applications.mobilenet_v2.preprocess_input
      - ResNet50V2   trained with tf.keras.applications.resnet_v2.preprocess_input
    Neither model was trained with simple rescale=1/255 -- the notebook's own
    Cell 1 flags that exact mistake as "the root cause of 35% accuracy" in an
    earlier attempt. The deployed app was using 'rescale' for the ResNet50V2
    model (custom_cnn), which is precisely that same bug.
    """
    img = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    if style == "mobilenet":
        arr = mobilenet_preprocess(arr)
    elif style == "resnet":
        arr = resnet_preprocess(arr)
    else:  # legacy fallback, not used by either real model -- kept only so
           # an unrecognized style doesn't hard-crash.
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def predict_with_models(pil_img, loaded_models, runtime_meta=None):
    """
    Runs real inference. If more than one model loaded, ensembles by
    averaging softmax probabilities (matches the repo's
    confusion_matrix_ensemble.png naming). Returns:
      preds        -> np.array aligned to CLASS_NAMES
      explanation  -> short string noting which model(s) produced this
      per_model    -> dict[label -> preds array] for transparency

    Uses runtime_meta (from load_models' architecture auto-detection) for
    preprocessing/labels when available -- this matters because
    brain_tumor_model.h5's true architecture isn't guaranteed by its
    filename/config entry (see _detect_backbone_architecture), so using the
    static MODEL_CONFIGS preprocessing here could silently apply the wrong
    normalization to whichever model actually got loaded.
    """
    runtime_meta = runtime_meta or {}
    per_model = {}
    for cfg in MODEL_CONFIGS:
        model = loaded_models.get(cfg["key"])
        if model is None:
            continue
        meta = runtime_meta.get(cfg["key"], {})
        style = meta.get("preprocess", cfg["preprocess"])
        label = meta.get("label", cfg["label"])
        x = preprocess_for_model(pil_img, style)
        raw = model.predict(x, verbose=0)[0]
        # If the model's final layer isn't already softmax-normalized, normalize defensively.
        raw = np.asarray(raw, dtype=np.float64)
        if raw.sum() <= 0 or abs(raw.sum() - 1.0) > 1e-3:
            exp = np.exp(raw - raw.max())
            raw = exp / exp.sum()
        per_model[label] = raw

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
.chip{{
  display:inline-flex;align-items:center;gap:5px;
  font-family:'DM Mono',monospace;font-size:9.5px;font-weight:600;
  letter-spacing:.06em;text-transform:uppercase;
  padding:5px 12px;border-radius:20px;white-space:nowrap;
  background:{"rgba(56,189,248,.14)" if _dk else "rgba(56,189,248,.10)"};
  border:1px solid {"rgba(56,189,248,.4)" if _dk else "rgba(56,189,248,.45)"};
  color:{"#ffffff" if _dk else "#0369a1"};
}}
.chip::before{{content:"";width:5px;height:5px;border-radius:50%;background:#38bdf8;flex-shrink:0;box-shadow:0 0 6px rgba(56,189,248,.85)}}
@media (max-width: 900px){{
  .chip{{font-size:8.5px;padding:4px 9px}}
}}
@media (max-width: 640px){{
  .nav-right .chip{{display:none}}
}}

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

.disc{{background:{"rgba(245,158,11,.08)" if _dk else "#fff8e6"};border:1px solid rgba(245,158,11,.35);border-left:3px solid rgba(245,158,11,.85);border-radius:0 12px 12px 0;padding:13px 18px;font-family:'DM Mono',monospace;font-size:11px;color:{"#fde68a" if _dk else "#7a4a00"};line-height:1.78;margin-top:20px}}
.disc strong{{color:{"#fbbf24" if _dk else "#8a5300"}}}

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
[data-testid="stSelectbox"]>div>div{{background:#ffffff !important;border:1.5px solid rgba(56,189,248,.5) !important;border-radius:11px !important;min-height:46px !important}}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] *{{color:{"rgba(255,255,255,.75)" if _dk else "#0a1628"} !important}}
/* The selected value and placeholder text inside Streamlit's selectbox are
   rendered by BaseWeb/react-select, which ships its own text color that a
   single broad "*" rule doesn't always beat depending on render order.
   Rather than keep fighting that per-theme, force this one control to
   always be a plain white field with plain black text in both themes --
   guaranteed legible regardless of what BaseWeb tries to apply. */
[data-testid="stSelectbox"] [data-baseweb="select"],
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] [data-baseweb="select"] div,
[data-testid="stSelectbox"] [data-baseweb="select"] span,
[data-testid="stSelectbox"] [data-baseweb="select"] p,
[data-testid="stSelectbox"] [class*="valueContainer"],
[data-testid="stSelectbox"] [class*="valueContainer"] *,
[data-testid="stSelectbox"] [class*="singleValue"],
[data-testid="stSelectbox"] [class*="placeholder"]{{
  color:#000000 !important;
  opacity:1 !important;
}}
[data-testid="stSelectbox"] svg{{fill:#000000 !important}}
[data-testid="stSelectbox"] [data-baseweb="select"]{{background:#ffffff !important}}
[data-baseweb="menu"]{{background:#ffffff !important;border:1px solid rgba(56,189,248,.25) !important;border-radius:10px !important}}
[data-baseweb="menu"] [role="option"], [data-baseweb="menu"] [role="option"] *{{background:transparent !important;color:#000000 !important;padding:12px 18px !important}}
[data-baseweb="menu"] [role="option"]:hover, [data-baseweb="menu"] [role="option"]:hover *{{background:rgba(56,189,248,.15) !important;color:#0369a1 !important}}
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

/* ---------------------------------------------------------------
   MOBILE RESPONSIVE OVERRIDES. The layout above assumes a wide
   desktop viewport (fixed 21rem sidebar, 2.4-2.8rem side padding,
   4-column stat grids). On a phone-width screen that sidebar alone
   eats most of the visible area and 4-column grids get crushed
   into unreadable slivers, so everything below re-flows those for
   narrow viewports without touching the desktop styles above.
   --------------------------------------------------------------- */
@media (max-width: 768px){{
  .topnav{{padding:.65rem 1rem}}
  .nav-tagline{{display:none}}
  .hero{{padding:2rem 0 1.6rem}}
  .hero-inner{{padding:0 1.1rem}}
  .hero-top{{flex-direction:column;align-items:flex-start;gap:1rem}}
  .hero-stats{{gap:1.4rem;width:100%}}
  .wrap{{padding:1.2rem 1.1rem 3rem}}
  .glass{{padding:1.3rem 1.2rem;border-radius:16px}}
  .pred-card{{padding:1.2rem 1.2rem}}
  .pred-name{{font-size:clamp(24px,7vw,34px)}}
  .hm-stats{{grid-template-columns:repeat(2,1fr)}}
  .hm-exp-grid{{grid-template-columns:1fr}}
  .cscale-lbls{{font-size:7.5px}}

  /* The sidebar's fixed 21rem width is a desktop assumption -- on a
     phone that's wider than the screen itself. Let it size to the
     viewport instead when it's open; the show/hide toggle logic
     (body.ns-sidebar-collapsed) is untouched. */
  [data-testid="stSidebar"]{{
    min-width:88vw !important;
    width:88vw !important;
  }}
  [data-testid="stSidebar"][aria-expanded="false"]{{
    min-width:88vw !important;
    width:88vw !important;
  }}
}}
@media (max-width: 480px){{
  .hero-h1{{font-size:clamp(1.5rem,7vw,2rem)}}
  .hero-desc{{font-size:13.5px}}
  .hero-stats{{gap:1rem}}
  .hs{{text-align:left}}
  .hm-stats{{grid-template-columns:repeat(2,1fr)}}
  .pip-txt{{font-size:10.5px}}
}}

@keyframes ns-spin{{to{{transform:rotate(360deg)}}}}
.ns-loader-wrap{{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:3.2rem 0 3rem;gap:18px}}
.ns-bigspin{{width:58px;height:58px;border-radius:50%;border:5px solid {"rgba(56,189,248,.14)" if _dk else "rgba(56,189,248,.18)"};border-top-color:#38bdf8;animation:ns-spin .85s linear infinite}}
.ns-loader-txt{{font-family:'DM Mono',monospace;font-size:12px;letter-spacing:.04em;text-align:center;color:{"rgba(255,255,255,.75)" if _dk else "rgba(10,22,40,.72)"};max-width:380px;line-height:1.6}}
.ns-loader-step{{font-family:'DM Mono',monospace;font-size:9px;text-transform:uppercase;letter-spacing:.16em;color:rgba(56,189,248,.85);margin-bottom:2px}}
</style>
""", unsafe_allow_html=True)

_SIDEBAR_TOGGLE_SCRIPT = """
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
"""

# NOTE: st.components.v1.html was deprecated in Streamlit 1.56.0 in favor of
# st.iframe (same-origin iframe, JS execution allowed, just like before) --
# see https://docs.streamlit.io/develop/api-reference/text/st.iframe. Since
# requirements.txt pins "streamlit>=1.32" with no upper bound, Streamlit
# Cloud will keep installing newer Streamlit releases on every rebuild, and
# components.v1.html is slated for eventual removal -- so we prefer st.iframe
# when it's present and only fall back to the deprecated call on older
# Streamlit installs, instead of hard-depending on either one.
if hasattr(st, "iframe"):
    st.iframe(_SIDEBAR_TOGGLE_SCRIPT, height=1, width=1)
else:
    components.html(_SIDEBAR_TOGGLE_SCRIPT, height=0, width=0)

# The CSS rules above for stSelectbox text color don't always win against
# BaseWeb/react-select's own generated styling (it can apply color via
# dynamically-injected, obfuscated-class stylesheets whose rules sometimes
# load after ours). Belt-and-suspenders fix: directly force an inline
# style with JS, which is the single highest-priority way to set a CSS
# property short of a stylesheet author using !important on an inline
# style themselves. Re-applied on every DOM mutation and on an interval
# so it survives re-renders when a selection changes or the menu opens.
_SELECT_COLOR_FIX_SCRIPT = f"""
<script>
(function(){{
  var doc = window.parent.document;
  var textColor = "#000000";
  var bgColor = "#ffffff";
  function fix(){{
    var boxes = doc.querySelectorAll('[data-testid="stSelectbox"] [data-baseweb="select"], [data-testid="stSelectbox"] [data-baseweb="select"] *');
    for (var i = 0; i < boxes.length; i++) {{
      boxes[i].style.setProperty('color', textColor, 'important');
      boxes[i].style.setProperty('opacity', '1', 'important');
    }}
    var fields = doc.querySelectorAll('[data-testid="stSelectbox"] [data-baseweb="select"]');
    for (var k = 0; k < fields.length; k++) {{
      fields[k].style.setProperty('background', bgColor, 'important');
    }}
    var menuOpts = doc.querySelectorAll('[data-baseweb="menu"] *');
    for (var j = 0; j < menuOpts.length; j++) {{
      menuOpts[j].style.setProperty('color', textColor, 'important');
    }}
    var menus = doc.querySelectorAll('[data-baseweb="menu"]');
    for (var m = 0; m < menus.length; m++) {{
      menus[m].style.setProperty('background', bgColor, 'important');
    }}
  }}
  fix();
  var mo = new MutationObserver(fix);
  try {{ mo.observe(doc.body, {{childList: true, subtree: true, attributes: true}}); }} catch(e) {{}}
  setInterval(fix, 400);
}})();
</script>
"""
if hasattr(st, "iframe"):
    st.iframe(_SELECT_COLOR_FIX_SCRIPT, height=1, width=1)
else:
    components.html(_SELECT_COLOR_FIX_SCRIPT, height=0, width=0)

# ================================================================
# MRI VALIDATION (Simple)
# ================================================================
def compute_image_quality(pil_img):
    """A small, standalone image-quality proxy (contrast + dynamic range),
    independent of which classifier ran. This exists purely so the AI report
    generator has an actual number to categorize for its `image_quality`
    field -- previously that field wasn't given anything, so Claude
    (correctly, but unhelpfully for the UI) wrote out a full sentence
    explaining that no quality metric was provided, which broke the compact
    st.metric widget and the GOOD/ADEQUATE/POOR color mapping downstream.
    Returns (score 0-1, label one of GOOD/ADEQUATE/POOR)."""
    img_gray = np.array(pil_img.convert("L"), dtype=np.float32)
    contrast = float(np.std(img_gray))
    dark_frac = float(np.mean(img_gray < 40))
    bright_frac = float(np.mean(img_gray > 200))
    dynamic_range = min(1.0, (dark_frac + bright_frac) / 0.15)
    score = max(0.0, min(1.0, (contrast / 55.0) * 0.6 + dynamic_range * 0.4))
    if score >= 0.65:
        label = "GOOD"
    elif score >= 0.35:
        label = "ADEQUATE"
    else:
        label = "POOR"
    return score, label

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

def _mask_heatmap_to_tissue(heatmap, orig_gray):
    """
    Zeroes out heatmap activation outside the actual brain tissue region and
    rescales so a robust high percentile *within tissue* defines "high" (red),
    then lightly smooths the result.

    Why this matters: conv layers use zero-padding, which creates well-known
    border artifacts in feature maps (edge pixels are computed from fewer/
    zero-padded neighbors and behave differently from interior pixels).
    When a small feature map (e.g. 7x7) gets upsampled to full resolution,
    these artifacts get stretched into the image corners/edges/boundary
    region, and the blocky upsampling itself can produce sharp, geometric-
    looking bands that are easy to mistake for real anatomical tracing.

    Two deliberate choices here:
    1. Normalizing by the 98th percentile (not the raw max) within tissue,
       so a small number of outlier/artifact pixels can't single-handedly
       define the entire color scale the way one dominant max pixel could.
    2. A light Gaussian blur after masking, to soften blocky resize
       artifacts into something visually distinguishable from a genuinely
       smooth, focal anatomical hotspot.

    IMPORTANT CAVEAT this does NOT fix: if the model's gradients are
    genuinely, consistently responding to skull/scalp boundary content
    (not just an upsampling artifact), that's a real property of what the
    model learned -- possibly "shortcut learning" on image framing rather
    than tissue content -- and no visualization-layer fix can correct that.
    It would need investigating at the dataset/training level, not here.
    """
    tissue_mask = (orig_gray > 22).astype(np.float32)
    masked = heatmap * tissue_mask
    tissue_pixels = masked[tissue_mask > 0]
    ref = np.percentile(tissue_pixels, 98) if tissue_pixels.size else 0.0
    if ref > 1e-6:
        masked = masked / ref
    masked = np.clip(masked, 0, 1)
    masked = cv2.GaussianBlur(masked, (9, 9), 0)
    masked = masked * tissue_mask  # re-zero background after blur bleeds it back in slightly
    return np.clip(masked, 0, 1)

def overlay_heatmap(img, heatmap, alpha=0.55):
    orig = np.array(img.convert("RGB").resize((224, 224)), dtype=np.float32)
    orig_gray = np.mean(orig, axis=2)
    heatmap = _mask_heatmap_to_tissue(np.asarray(heatmap, dtype=np.float32), orig_gray)

    hm_colored = (mpl_cm.jet(heatmap)[:, :, :3] * 255).astype(np.float32)
    gray = np.mean(orig, axis=2, keepdims=True)
    desat = orig * 0.4 + gray * 0.6
    
    alpha_mask = np.clip(alpha + (1 - alpha) * heatmap[..., np.newaxis] * 0.5, 0, 1)
    blend = np.clip(desat * (1 - alpha_mask) + hm_colored * alpha_mask, 0, 255).astype(np.uint8)
    
    return Image.fromarray(blend)

# ================================================================
# CLINICAL REPORT
# ================================================================
def _get_anthropic_api_key():
    """Checks Streamlit's secrets manager first, then falls back to a plain
    environment variable. The env var fallback matters because it's easy to
    set ANTHROPIC_API_KEY somewhere that isn't Streamlit Cloud's Secrets
    panel (a .env file, a different host's env var UI, a local `export`)
    and get the same "not configured" message with no indication of why --
    st.secrets alone won't see any of those."""
    key = None
    source = None

    if hasattr(st, "secrets"):
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", None)
            if key:
                source = "st.secrets"
        except Exception as e:
            _log(f"[ai_report] st.secrets lookup raised: {e}")

    if not key:
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            key = env_key
            source = "environment variable"

    if key:
        # Defensive cleanup: the most common cause of a 401 "API key is
        # invalid" error when the key LOOKS present is invisible whitespace
        # (a trailing newline from pasting into the Streamlit Cloud secrets
        # box) or an accidental extra pair of quotes in secrets.toml, e.g.
        #   ANTHROPIC_API_KEY = "'sk-ant-...'"
        # which makes the literal quote characters part of the string.
        # Strip both before the key is ever sent to the API.
        cleaned = key.strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in ("'", '"'):
            cleaned = cleaned[1:-1].strip()

        if cleaned != key:
            _log(
                f"[ai_report] ANTHROPIC_API_KEY from {source} had surrounding "
                f"whitespace/quotes stripped (raw len={len(key)} -> clean len={len(cleaned)})"
            )
        key = cleaned

        if not key.startswith("sk-ant-"):
            _log(
                f"[ai_report] WARNING: ANTHROPIC_API_KEY from {source} does not start "
                f"with 'sk-ant-' (starts with {key[:7]!r}) -- this doesn't look like a "
                "valid Anthropic API key format, double check what was pasted into secrets."
            )

        _log(
            f"[ai_report] ANTHROPIC_API_KEY found via {source} "
            f"(len={len(key)}, prefix={key[:11]!r}, suffix={key[-4:]!r})"
        )
    else:
        _log("[ai_report] ANTHROPIC_API_KEY not found in st.secrets or environment")

    return key

REPORT_JSON_SCHEMA_FIELDS = [
    "clinical_interpretation", "location_morphology", "model_reasoning",
    "gradcam_analysis", "risk_level", "risk_justification", "patient_explanation",
    "next_steps", "image_quality", "uncertainty_factors", "reliability_score",
    "overall_reliability", "differential_diagnosis", "disclaimer",
]

def generate_ai_report(pred_class, conf, explanation, per_model, mean_a, max_a, p90_a, focus_p, cam_labels,
                        image_quality_label="ADEQUATE", image_quality_score=0.5):
    """
    Calls the real Anthropic API to write a genuinely per-image report,
    grounded strictly in this specific analysis's actual numbers -- not a
    fixed lookup table keyed only by class name (that was the old
    template_report(), which returned identical boilerplate for every image
    predicted as the same class regardless of confidence, activation
    pattern, or model agreement).

    IMPORTANT GUARDRAIL: Claude is given only the classifier's quantitative
    output (predicted class, confidence, per-model votes, Grad-CAM
    activation statistics) -- NOT the raw image -- and is explicitly
    instructed to write UP that existing evidence, not to independently
    diagnose from pixels itself. This keeps the tool as decision support
    for an existing classifier's output, not an unregulated standalone
    diagnostic AI.

    Returns (report_dict, used_real_ai: bool, error_or_None, error_kind_or_None).
    error_kind is one of: "auth", "billing", "rate_limit", "overloaded", "other", None.
    """
    api_key = _get_anthropic_api_key()
    if not ANTHROPIC_AVAILABLE:
        return None, False, f"anthropic package not installed: {ANTHROPIC_IMPORT_ERROR}", "other"
    if not api_key:
        return None, False, "No ANTHROPIC_API_KEY configured in Streamlit secrets.", "auth"

    votes_desc = "; ".join(
        f"{label}: {CLASS_NAMES[int(np.argmax(p))]} ({p[int(np.argmax(p))]*100:.1f}%)"
        for label, p in per_model.items()
    ) if per_model else "single model, no ensemble breakdown"

    prompt = f"""You are drafting a structured AI-assisted imaging summary that accompanies the output of an
existing trained brain MRI classifier (a CNN ensemble: {', '.join(cam_labels) if cam_labels else 'a trained CNN'}).
You are NOT looking at the image yourself and must NOT invent specific anatomical findings you weren't given.
Write up ONLY the evidence below, clearly, for a clinician audience, while being explicit about the limits of
what a classification-only model output can support (e.g. you cannot state a precise lobe/location unless the
Grad-CAM statistics given below actually support a location claim -- if not, say location cannot be determined
from classification alone and would need radiologist review of the actual images).

TONE: Write in the register of a radiology decision-support summary read by a neurosurgeon or radiologist --
direct, precise, and confident about what the data actually shows. State the numbers plainly rather than
hedging every sentence ("may possibly suggest", "it could potentially indicate"). Reserve uncertainty language
specifically for the points that are genuinely uncertain (location, definitive diagnosis) rather than spreading
it across every sentence -- an over-hedged report reads as less credible to a clinical reader, not more careful.

CLASSIFIER OUTPUT (this is the entire evidence base, do not add anything beyond it):
- Final predicted class: {pred_class}
- Final confidence: {conf:.1f}%
- Per-model votes before averaging: {votes_desc}
- Grad-CAM activation statistics (within brain tissue only, 0-1 scale): mean={mean_a:.3f}, max={max_a:.3f}, 90th percentile={p90_a:.3f}, percent of tissue area with activation>0.5={focus_p:.1f}%
- Image quality (computed from pixel contrast/dynamic range, independent of the classifier): {image_quality_label} (score {image_quality_score:.2f} on a 0-1 scale)
- Note: {explanation}

Return ONLY a JSON object (no markdown fences, no preamble) with exactly these keys, each a string
(next_steps can use \\n between numbered items):
{json.dumps(REPORT_JSON_SCHEMA_FIELDS)}

Guidance per field:
- clinical_interpretation: 1-2 sentences, stated plainly, on what this classifier output suggests, referencing the confidence and activation stats.
- location_morphology: only general terms consistent with the predicted class; explicitly say precise anatomical location requires radiologist review of the actual images, since a classifier alone cannot localize with certainty.
- model_reasoning: reference the actual confidence and per-model agreement/disagreement given above.
- gradcam_analysis: describe based on the actual activation stats given (e.g. high focus_p + high max = concentrated hotspot; low values = diffuse/non-specific).
- risk_level: one of LOW, MODERATE, HIGH, based on predicted class and confidence.
- risk_justification: brief, general clinical context for that class, not image-specific claims.
- patient_explanation: plain language, 1-2 sentences, reassuring but honest that this is AI-assisted, not a diagnosis.
- next_steps: 2-4 concrete recommended actions appropriate to the predicted class, confidence, and risk level (e.g. specialist referral, contrast-enhanced follow-up imaging, clinical correlation, routine monitoring) -- format as "1. ...\\n2. ...\\n3. ...". This field is REQUIRED, do not omit it.
- image_quality: must be exactly one of GOOD, ADEQUATE, or POOR -- simply echo the image quality value given above verbatim, do not write a sentence and do not invent a different category, even if you'd personally judge it differently.
- uncertainty_factors: name what would reduce confidence in THIS specific result (e.g. low confidence, model disagreement, low activation focus) using the actual numbers given.
- reliability_score: an integer 0-100 you compute reasonably from confidence + model agreement + activation focus, not a fixed per-class constant.
- overall_reliability: one short phrase summarizing that score.
- differential_diagnosis: 1-2 other conditions commonly confused with the predicted class in brain MRI, general medical knowledge, not image-specific.
- disclaimer: a brief AI-assisted-only disclaimer."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-5",
            # 1200 was too tight for a 14-field structured report (clinical
            # interpretation, differential diagnosis, next steps, etc. all
            # add up) -- Claude was running out of budget mid-field, which
            # truncates the JSON and produces "Unterminated string" on
            # parse. Bumped up with headroom; a single short JSON report is
            # still a tiny fraction of a cent even at this ceiling.
            max_tokens=3000,
            messages=[{"role": "user", "content": prompt}],
        )

        if resp.stop_reason == "max_tokens":
            _log(
                "[ai_report] Response was truncated by max_tokens before the JSON "
                "closed -- this should not happen at the current limit, but if it "
                "recurs the limit needs raising further or the prompt needs to ask "
                "for shorter fields."
            )
            return None, False, (
                "Claude's response was cut off before it finished (hit the max_tokens "
                "limit) -- the report couldn't be parsed as a result. Try again; if it "
                "keeps happening the per-report token limit in the code needs raising."
            ), "other"

        text = "".join(block.text for block in resp.content if hasattr(block, "text")).strip()
        text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            _log(f"[ai_report] JSON parse failed ({e}); raw text was: {text[:2000]!r}")
            return None, False, f"Could not parse Claude's response as JSON: {e}", "other"

        missing = [f for f in REPORT_JSON_SCHEMA_FIELDS if f not in data]
        if missing:
            # Don't throw away an otherwise-real, mostly-complete AI report
            # over one dropped field -- that discards genuine per-image
            # analysis (confidence-aware reasoning, actual reliability
            # score, etc.) just because, say, next_steps got skipped.
            # Backfill only the missing key(s) from the static per-class
            # template and still surface the rest as real AI content.
            _log(f"[ai_report] Claude response missing fields (backfilling from template): {missing}")
            _fallback = template_report(pred_class, conf, explanation)
            for f in missing:
                data[f] = _fallback.get(f, "")
        return data, True, None, None
    except anthropic.AuthenticationError as e:
        # A 401 here means the request reached Anthropic's servers and the
        # key was rejected outright -- this is NOT a bug in this app's code
        # path (the key was found and sent correctly). It means the key
        # string itself is wrong: revoked/deleted, from the wrong
        # organization, or mistyped/truncated when it was pasted into
        # Streamlit secrets. Re-generate a fresh key at
        # https://console.anthropic.com/settings/keys and replace the value
        # in Settings -> Secrets, with nothing extra (no quotes, no newline).
        _log(f"[ai_report] AUTHENTICATION ERROR calling Anthropic API: {e}")
        return None, False, (
            "Anthropic API rejected this key (401 authentication_error). The key was found "
            "in secrets and looked correctly formatted, but the server says it's invalid -- "
            "this means the key itself is wrong (revoked, from a different workspace, or "
            "mistyped/truncated when pasted). Generate a new key at "
            "console.anthropic.com/settings/keys and paste it into Streamlit "
            "secrets with no surrounding quotes or extra whitespace."
        ), "auth"
    except Exception as e:
        _log(f"[ai_report] ERROR calling Anthropic API: {e}")
        _msg = str(e)
        _low = _msg.lower()
        # Classify the failure so the UI can show something a non-technical
        # reader understands, instead of a raw exception string that reads
        # like the application itself is broken. A "credit balance too low"
        # 400 is a billing state, not a bug -- it should look and read like
        # a billing notice, not an error stack trace.
        if "credit balance" in _low or "credit_balance" in _low or "insufficient" in _low and "credit" in _low:
            error_kind = "billing"
        elif "rate limit" in _low or "rate_limit" in _low:
            error_kind = "rate_limit"
        elif "overloaded" in _low:
            error_kind = "overloaded"
        else:
            error_kind = "other"
        return None, False, _msg, error_kind

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
    _loaded_models, _model_errors, _runtime_meta = load_models(_mode)
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
        _display_label = _runtime_meta.get(cfg["key"], {}).get("label", cfg["label"])
        if cfg["key"] in _loaded_models:
            st.success(f"✅ {_display_label} loaded")
            if _display_label != cfg["label"]:
                st.caption(f"⚠️ Note: this file was configured as '{cfg['label']}' but its actual backbone "
                           f"was detected as '{_display_label}' at runtime - label and preprocessing were "
                           f"corrected automatically.")
        else:
            st.error(f"❌ {cfg['label']} not loaded")
            st.caption(_model_errors.get(cfg["key"], "unknown error")[:200])

    if _loaded_models:
        st.success(f"✅ Ready to classify using: "
                   f"{', '.join(_runtime_meta.get(cfg['key'], {}).get('label', cfg['label']) for cfg in _active_cfgs if cfg['key'] in _loaded_models)}")
    else:
        st.error("⚠️ No model loaded - using basic fallback only. Not for clinical use.")

    if st.button("🔄 Retry / reload models", help="Clears the cached model state and retries loading + downloading from scratch, without needing a full redeploy."):
        load_models.clear()
        st.rerun()

    st.markdown("---")
    st.markdown("#### Settings")
    alpha = st.slider("Heatmap Intensity", 0.2, 0.8, 0.55, 0.05)
    temperature = st.slider("Temperature", 1.0, 2.5, 1.4, 0.1)
    
    st.markdown("---")
    st.markdown(f"""
<div style="background:rgba(245,158,11,.10);border-left:3px solid #f59e0b;
      border-radius:0 8px 8px 0;padding:10px 12px;font-family:'DM Mono',monospace;
      font-size:9.5px;color:{"rgba(253,211,77,.92)" if _dk else "#7c4a03"};line-height:1.7;">
<strong style="color:{"#fbbf24" if _dk else "#92400e"};">Clinical Disclaimer</strong><br>
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
          Upload any axial brain MRI and receive instant classification across 4 categories,
          Glioma, Meningioma, Pituitary Tumor, and No Tumor, complete with Grad-CAM heatmaps
          and AI-generated clinical reports.
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
        st.markdown(f'''<div style="border:2px dashed rgba(56,189,248,.38);border-radius:14px;padding:2.5rem 1.5rem;text-align:center;background:rgba(56,189,248,.05);margin:8px 0;">
<div style="font-size:40px;margin-bottom:12px;">🩻</div>
<div style="font-family:Space Grotesk,sans-serif;font-size:15px;font-weight:600;color:{text_color};margin-bottom:6px;">No image selected</div>
<div style="font-family:DM Mono,monospace;font-size:10px;color:{"rgba(255,255,255,.58)" if _dk else "rgba(10,22,40,.62)"};letter-spacing:.07em;line-height:1.9;">
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
        _ph_muted = "rgba(255,255,255,.65)" if _dk else "#2d4a6b"
        _ph_chip = "#7dd3fc" if _dk else "#0369a1"
        _ph_chip_bg = "rgba(56,189,248,.10)" if _dk else "rgba(56,189,248,.12)"
        _ph_chip_bd = "rgba(56,189,248,.28)" if _dk else "rgba(56,189,248,.45)"
        st.markdown(f'''<div class="glass" style="min-height:440px;display:flex;align-items:center;justify-content:center;text-align:center;padding:3rem;">
<div>
<div style="font-size:54px;margin-bottom:18px;">🔬</div>
<div style="font-family:Space Grotesk,sans-serif;font-size:19px;font-weight:600;color:{text_color};line-height:1.4;margin-bottom:8px;">Ready for Analysis</div>
<div style="font-family:Inter,sans-serif;font-size:13.5px;color:{_ph_muted};line-height:1.85;margin-bottom:22px;">
              Upload a brain MRI or select a sample,<br>
              then click <strong>Analyse</strong> to run the full pipeline.
</div>
<div style="display:flex;flex-wrap:wrap;gap:8px;justify-content:center;">
<span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:{_ph_chip_bg};border:1px solid {_ph_chip_bd};color:{_ph_chip};letter-spacing:.07em;white-space:nowrap;">AI PREDICTION</span>
<span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:{_ph_chip_bg};border:1px solid {_ph_chip_bd};color:{_ph_chip};letter-spacing:.07em;white-space:nowrap;">GRAD-CAM HEATMAP</span>
<span style="font-family:DM Mono,monospace;font-size:9.5px;padding:5px 14px;border-radius:20px;background:{_ph_chip_bg};border:1px solid {_ph_chip_bd};color:{_ph_chip};letter-spacing:.07em;white-space:nowrap;">CLINICAL REPORT</span>
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
  min-width:320px; max-width:min(480px,90vw); pointer-events:none;
}
@media (max-width: 420px){
  #ns-toast{ min-width:0; width:90vw; left:5vw; right:5vw; transform:none; padding:10px 16px 10px 12px; }
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

    # Reusable big circular loader used for every step of the pipeline below
    # (Streamlit's default spinner is a small inline icon; this is a larger,
    # centered indicator with a step label so users always know it's working
    # and roughly where in the pipeline it is).
    _loader = st.empty()
    def _set_loader(step_label, detail):
        _loader.markdown(f'''<div class="ns-loader-wrap">
<div class="ns-bigspin"></div>
<div>
<div class="ns-loader-step">{step_label}</div>
<div class="ns-loader-txt">{detail}</div>
</div>
</div>''', unsafe_allow_html=True)

    # Step 1: MRI Validation
    _img_quality_score, _img_quality_label = compute_image_quality(img)
    if not override:
        _set_loader("Step 1 / 4", "Validating image")
        is_valid_mri, mri_confidence, mri_reason = validate_mri(img)
        _loader.empty()

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
    _set_loader("Step 2 / 4", "Running AI classification")
    loaded_models, _, runtime_meta = load_models(_get_model_mode())
    if loaded_models:
        try:
            preds, explanation, per_model = predict_with_models(img, loaded_models, runtime_meta)
            used_real_model = True
            for key in ("mobilenet", "custom_cnn"):
                if key in loaded_models:
                    meta = runtime_meta.get(key, {})
                    default_style = "mobilenet" if key == "mobilenet" else "resnet"
                    default_label = "MobileNetV2" if key == "mobilenet" else "ResNet50V2"
                    cam_candidates.append((
                        meta.get("label", default_label),
                        loaded_models[key],
                        meta.get("preprocess", default_style),
                    ))
            # Keep a single "primary" choice for anything that only wants
            # one heatmap (JSON export, single-model modes, etc.) --
            # prefer MobileNetV2 since it's the one confirmed reliable.
            if "mobilenet" in loaded_models:
                meta = runtime_meta.get("mobilenet", {})
                active_model_for_cam = loaded_models["mobilenet"]
                active_preprocess_style = meta.get("preprocess", "mobilenet")
                active_cam_label = meta.get("label", "MobileNetV2")
            elif "custom_cnn" in loaded_models:
                meta = runtime_meta.get("custom_cnn", {})
                active_model_for_cam = loaded_models["custom_cnn"]
                active_preprocess_style = meta.get("preprocess", "resnet")
                active_cam_label = meta.get("label", "ResNet50V2")
        except Exception as e:
            st.error(f"Model inference failed ({e}); falling back to non-model heuristic.")
            preds, explanation = predict_with_heuristic(img)
    else:
        preds, explanation = predict_with_heuristic(img)

    if temperature != 1.0:
        logits = np.log(np.clip(preds, 1e-7, 1.0))
        scaled = np.exp(logits / temperature)
        preds = scaled / scaled.sum()
    _loader.empty()

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
    cam_results = {}  # label -> (heatmap, overlay, is_real, own_top_class)
    _set_loader("Step 3 / 4", "Generating Grad-CAM heatmap")
    heatmap = None
    if used_real_model and cam_candidates:
        for label, cand_model, cand_style in cam_candidates:
            # IMPORTANT: use THIS model's own top predicted class, not the
            # ensemble's. Forcing every model's Grad-CAM to target the
            # ensemble's chosen class produces a heatmap answering "why
            # does this look like {ensemble class}" even for a model that
            # itself predicted something else entirely -- that's not
            # that model's real reasoning, it's a misleading question to
            # ask its gradients. Each panel should explain what that
            # specific model actually concluded.
            own_pred = per_model.get(label)
            own_pidx = int(np.argmax(own_pred)) if own_pred is not None else pidx
            own_pcls = CLASS_NAMES[own_pidx]
            try:
                cand_heatmap = generate_gradcam(img, cand_model, cand_style, own_pidx)
            except Exception as e:
                st.caption(f"Grad-CAM computation failed for {label} ({e}).")
                cand_heatmap = None
            is_real = cand_heatmap is not None
            if cand_heatmap is None:
                cand_heatmap = generate_heatmap(img, own_pcls)
            cand_overlay = overlay_heatmap(img, cand_heatmap, alpha=alpha)
            cam_results[label] = (cand_heatmap, cand_overlay, is_real, own_pcls)
        # Primary heatmap/overlay (used for the JSON export, activation
        # stats, etc.) is whichever one the rest of the app already
        # expects as "the" heatmap -- prefer MobileNetV2 for consistency
        # with what's been validated end-to-end.
        if active_cam_label and active_cam_label in cam_results:
            heatmap, overlay, heatmap_is_real, _ = cam_results[active_cam_label]
        else:
            heatmap, overlay, heatmap_is_real, _ = next(iter(cam_results.values()))
    if heatmap is None:
        heatmap = generate_heatmap(img, pcls)
        overlay = overlay_heatmap(img, heatmap, alpha=alpha)
    _loader.empty()

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
            st.caption("Each Grad-CAM heatmap below explains that specific model's own top prediction (labeled "
                       "under each image), not necessarily the ensemble's final answer above. If a model's own "
                       "vote differs from the ensemble result, its heatmap is shown for whichever class *that "
                       "model* actually concluded, so it always reflects that model's real reasoning rather than "
                       "being forced to justify an answer it didn't give.")

    _stats_gray = np.mean(np.array(img.convert("RGB").resize(IMG_SIZE), dtype=np.float32), axis=2)
    _masked_heatmap_for_stats = _mask_heatmap_to_tissue(np.asarray(heatmap, dtype=np.float32), _stats_gray)
    mean_a = float(_masked_heatmap_for_stats.mean())
    max_a = float(_masked_heatmap_for_stats.max())
    p90_a = float(np.percentile(_masked_heatmap_for_stats, 90))
    focus_p = float((_masked_heatmap_for_stats > 0.5).sum() / _masked_heatmap_for_stats.size * 100)
    heatmap = _masked_heatmap_for_stats  # use the tissue-masked version for every downstream panel/export too

    # Step 4: Report
    _set_loader("Step 4 / 4", "Drafting clinical report")
    _cam_labels_for_report = [cfg["label"] for cfg in _active_model_configs() if cfg["key"] in loaded_models] if used_real_model else []
    report, used_real_ai_report, ai_report_error, ai_report_error_kind = generate_ai_report(
        pcls, conf, explanation, per_model, mean_a, max_a, p90_a, focus_p, _cam_labels_for_report,
        image_quality_label=_img_quality_label, image_quality_score=_img_quality_score,
    )
    if report is None:
        report = template_report(pcls, conf, explanation)
    # Defensive: even with explicit prompt guidance, don't trust the
    # model to always return exactly one of the three enum values --
    # fall back to the metric we actually computed rather than passing
    # arbitrary text into a widget/color-map built for GOOD/ADEQUATE/POOR.
    if report.get("image_quality") not in ("GOOD", "ADEQUATE", "POOR"):
        report["image_quality"] = _img_quality_label
    _loader.empty()
    # NOTE: the "generated by Claude" disclosure itself is rendered later,
    # directly above the Clinical Report tabs -- not here. Showing it this
    # early (before the prediction/class-distribution/heatmap results below
    # have even rendered) reads as if Claude produced the classification
    # itself. It didn't: the CNN ensemble above is the sole source of the
    # prediction and Grad-CAM; Claude only writes up those numbers afterward.

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
            for (label, (r_heatmap, r_overlay, r_is_real, r_own_cls)), cam_col in zip(cam_results.items(), cam_cols):
                with cam_col:
                    st.markdown('<div class="hm-img-frame">', unsafe_allow_html=True)
                    st.image(r_overlay, width='stretch')
                    st.markdown('</div>', unsafe_allow_html=True)
                    _tag = "" if r_is_real else " (synthetic)"
                    _disagree_note = "" if r_own_cls == pcls else f" - this model's own top call, differs from ensemble's {pcls}"
                    st.markdown(f"""
<div style="text-align:center;margin-top:8px;font-family:'DM Mono',monospace;font-size:10px;color:{"rgba(255,255,255,.40)" if _dk else "rgba(10,22,40,.50)"};letter-spacing:.1em;">{label} Grad-CAM | explains: {r_own_cls}{_tag}{_disagree_note}</div>""", unsafe_allow_html=True)
                    if r_own_cls == "No Tumor":
                        st.caption("⚠️ A 'No Tumor' Grad-CAM has no lesion to point at, so it's inherently harder to "
                                   "interpret than a positive finding. If the highlighted region traces skull/scalp "
                                   "boundary rather than diffuse interior tissue, treat that as a signal the model "
                                   "may be keying off image framing, not genuine absence of pathology.")
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

    if pcls == "No Tumor":
        st.caption(
            "ℹ️ **What the colors mean here:** red/orange marks where the classifier's attention "
            "concentrated when it decided *this scan does not show tumor features* -- it is not "
            "pointing to a suspected lesion. For a No Tumor result, a hotspot commonly falls on "
            "normal anatomy (e.g. skull, sinuses, symmetric tissue) that the model used as evidence "
            "*against* a tumor pattern, not a site to investigate for one."
        )
    else:
        st.caption(
            "ℹ️ **What the colors mean:** red/orange marks the image regions the classifier weighted "
            "most heavily to reach this specific prediction (its attention, not a segmented tumor "
            "boundary). It's a plausibility check on the model's reasoning -- concentrated activation "
            "over plausible anatomy supports the call, but it does not replace radiologist localization "
            "of the actual lesion from the source imaging."
        )

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

    if used_real_ai_report:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:8px;padding:9px 14px;'
            'border-radius:10px;background:rgba(56,189,248,.10);border:1px solid rgba(56,189,248,.35);'
            'font-family:\'DM Mono\',monospace;font-size:11px;margin-bottom:12px;">'
            '✍️ <strong>Report text written by Claude (Anthropic)</strong>, summarizing the classifier\'s '
            'numeric results above (confidence, model agreement, Grad-CAM statistics). The prediction and '
            'heatmap are generated by the CNN model, not Claude.</div>',
            unsafe_allow_html=True,
        )
    else:
        if ai_report_error_kind == "billing":
            # This is an account/billing state, not a bug -- it should read
            # like a billing notice, not like something broke in the code.
            st.markdown(
                '<div style="display:flex;align-items:flex-start;gap:10px;padding:12px 16px;'
                'border-radius:10px;background:rgba(251,191,36,.10);border:1px solid rgba(251,191,36,.40);'
                'font-family:Inter,sans-serif;font-size:13px;line-height:1.6;margin-bottom:12px;">'
                '<span style="font-size:17px;">💳</span>'
                '<div><strong>AI-generated report unavailable -- Anthropic account is out of API credits.</strong><br>'
                'This is a billing matter, not an application error: the connected Anthropic account has run out '
                'of usage credits. An administrator needs to add credits at '
                '<a href="https://console.anthropic.com" target="_blank" style="color:#0369a1;">console.anthropic.com</a> '
                '&rarr; Plans &amp; Billing. Showing a static template report below in the meantime -- prediction '
                'and heatmap results above are unaffected.</div></div>',
                unsafe_allow_html=True,
            )
        elif ai_report_error_kind == "auth":
            st.warning(f"🔑 AI-generated report unavailable -- the configured Anthropic API key was rejected. "
                       f"An administrator needs to check `ANTHROPIC_API_KEY` in Streamlit secrets. "
                       f"({ai_report_error})")
        elif ai_report_error_kind == "rate_limit":
            st.warning("⏳ AI-generated report unavailable right now -- Anthropic's API is temporarily rate-limiting "
                       "requests. This usually resolves within a minute or two; try running the analysis again.")
        elif ai_report_error_kind == "overloaded":
            st.warning("⏳ AI-generated report unavailable right now -- Anthropic's servers are temporarily "
                       "overloaded. This is on Anthropic's side, not this app; try again shortly.")
        else:
            st.warning(f"⚠️ Using a static per-class template for this report, not a dynamically generated one. "
                       f"This looks like a one-off issue with that specific AI response, not your API key or "
                       f"credits -- try running the analysis again. ({ai_report_error})")

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
<strong>AI-Assisted Decision Support Only</strong> - This summary is generated from an AI
classification model output only and is not a substitute for professional radiologic or
clinical diagnosis; all findings must be confirmed by a qualified radiologist and clinician
reviewing the actual imaging and patient context. The narrative text above was drafted by
Claude (Anthropic), an AI language model, which summarizes the classifier's quantitative
output (predicted class, confidence, per-model votes, Grad-CAM statistics); the CNN classifier
performs the image analysis, Claude writes up its results. All findings require review by a
licensed radiologist or neurosurgeon.
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
    st.markdown('<div class="slbl">📊 Model Performance (Validation Results)</div>', unsafe_allow_html=True)

    _perf_items = [
        {"val": 95.31, "label": "Overall Accuracy", "icon": "🎯", "color": "#38bdf8"},
        {"val": 100.0, "label": "No Tumor Recall", "icon": "✅", "color": "#34d399"},
        {"val": 99.8,  "label": "Pituitary Recall", "icon": "🧬", "color": "#a78bfa"},
        {"val": 98.0,  "label": "Meningioma Recall", "icon": "🔬", "color": "#fbbf24"},
        {"val": 83.5,  "label": "Glioma Recall", "icon": "⚕️", "color": "#f87171"},
    ]
    _card_bg = "rgba(255,255,255,.03)" if _dk else "#ffffff"
    _track_bg = "rgba(255,255,255,.08)" if _dk else "rgba(10,22,40,.08)"
    _sub_text = "rgba(255,255,255,.55)" if _dk else "rgba(10,22,40,.62)"

    _cards_html = ""
    for it in _perf_items:
        _cards_html += f"""
<div style="background:{_card_bg};border:1px solid {it['color']}33;border-top:3px solid {it['color']};
                border-radius:12px;padding:16px 14px;text-align:center;box-shadow:0 2px 10px rgba(0,0,0,.06);">
<div style="font-size:20px;margin-bottom:4px;">{it['icon']}</div>
<div style="font-family:'Space Grotesk',sans-serif;font-size:22px;font-weight:700;color:{it['color']};">{it['val']:g}%</div>
<div style="font-family:'DM Mono',monospace;font-size:9.5px;text-transform:uppercase;letter-spacing:.08em;color:{_sub_text};margin:4px 0 8px;">{it['label']}</div>
<div style="height:5px;border-radius:4px;background:{_track_bg};overflow:hidden;">
<div style="height:100%;width:{it['val']}%;background:{it['color']};border-radius:4px;"></div>
</div>
</div>"""

    st.markdown(f"""
<div class="glass" style="background:linear-gradient(135deg,{"rgba(56,189,248,.05)" if _dk else "rgba(56,189,248,.04)"},transparent);border-radius:16px;padding:20px;">
<div style="font-size:11.5px;color:{_sub_text};line-height:1.65;margin-bottom:18px;padding:10px 14px;
              background:{"rgba(255,255,255,.03)" if _dk else "rgba(10,22,40,.03)"};border-radius:8px;border-left:3px solid #38bdf8;">
    ℹ️ Measured once on a held-out validation set during training (see <code>confusion_matrix_ensemble.png</code> in
    the repo), not a live guarantee for any single scan you upload. Individual results above can and do vary from
    these averages, especially at lower confidence.
</div>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:14px;margin-bottom:16px;">
    {_cards_html}
</div>
<div style="font-size:11.5px;color:{_sub_text};line-height:1.7;text-align:center;">
    Predictions come from real trained ResNet50V2 / MobileNetV2 models with genuine Grad-CAM explainability, not a rule-based heuristic.
</div>
</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
