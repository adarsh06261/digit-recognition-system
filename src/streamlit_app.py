# src/streamlit_app.py

import sys, os
# ensure we can import sibling modules (predict_cnn, predict_rf, utils)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import importlib

st.set_page_config(page_title="Handwritten Digit Recognizer", layout="centered")

# --------------------------------------------------------------------
# Safe import helper
# --------------------------------------------------------------------
def try_import(module_name, func_candidates=None):
    try:
        mod = importlib.import_module(module_name)
        if not func_candidates:
            return mod, None
        for fn in func_candidates:
            if hasattr(mod, fn):
                return mod, getattr(mod, fn)
        return mod, None
    except Exception as e:
        return None, None

# Try loading modules
utils_mod, _ = try_import("utils")
utils_preprocess = getattr(utils_mod, "preprocess", None) if utils_mod else None

cnn_mod, cnn_predict = try_import("predict_cnn", ["predict_digit_cnn", "predict_cnn", "predict"])
rf_mod, rf_predict = try_import("predict_rf", ["predict_digit_rf", "predict_rf", "predict"])

# --------------------------------------------------------------------
# UI
# --------------------------------------------------------------------
st.title("🖊️ Handwritten Digit Recognizer")
st.caption("Draw a digit (0–9) below, choose a model, and click Predict.")

# Debug info in sidebar
with st.sidebar:
    st.header("Debug info")
    st.write("utils.py:", "✅" if utils_mod else "❌")
    st.write("predict_cnn.py:", "✅" if cnn_mod else "❌")
    st.write("predict_rf.py:", "✅" if rf_mod else "❌")

stroke_width = st.sidebar.slider("Brush thickness", 6, 30, 12, 1)

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#FFFFFF",       # white ink
    background_color="#000000",   # black background
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# --------------------------------------------------------------------
# Preprocess helper
# --------------------------------------------------------------------
def preprocess_canvas(img_rgba: np.ndarray) -> np.ndarray:
    """Return a (28,28) float32 array scaled to [0,1]."""
    if utils_preprocess is not None:
        try:
            return utils_preprocess(img_rgba)
        except Exception:
            pass
    pil = Image.fromarray(img_rgba.astype("uint8")).convert("L")
    pil = pil.resize((28, 28), Image.LANCZOS)
    pil = ImageOps.invert(pil)  # invert to MNIST style
    arr = np.array(pil).astype("float32") / 255.0
    return arr

# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------
model = st.radio("Model", ["CNN (high accuracy)", "Random Forest"], index=0, horizontal=True)

if st.button("Predict", type="primary", use_container_width=True):
    if canvas.image_data is None:
        st.warning("Please draw a digit first.")
    else:
        x = preprocess_canvas(canvas.image_data)
        try:
            if model.startswith("CNN"):
                if not cnn_predict:
                    raise RuntimeError("No CNN predictor found in predict_cnn.py")
                pred, probs = cnn_predict(x)
            else:
                if not rf_predict:
                    raise RuntimeError("No RF predictor found in predict_rf.py")
                pred, probs = rf_predict(x)

            st.subheader(f"Prediction: {pred}")

            # visualize probabilities if available
            import matplotlib.pyplot as plt
            labels = [str(i) for i in range(10)]
            y = None
            if isinstance(probs, dict):
                y = [float(probs.get(i, probs.get(str(i), 0.0))) for i in range(10)]
            else:
                try:
                    y = list(map(float, probs))
                except Exception:
                    y = None

            if y and len(y) == 10:
                fig, ax = plt.subplots()
                ax.bar(labels, y)
                ax.set_xlabel("Digit")
                ax.set_ylabel("Probability")
                ax.set_title("Class Probabilities")
                st.pyplot(fig)
            else:
                st.json(probs)

        except Exception as e:
            st.error(f"Prediction failed: {e}")

st.caption("💡 Tip: We resize drawings to 28×28 and invert colors to match MNIST. Use thicker strokes for better recognition.")
