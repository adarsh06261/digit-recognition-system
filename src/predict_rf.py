import numpy as np
import joblib
from PIL import Image
from src.utils import preprocess_pil_for_models

# Lazy-loaded singleton RF model
_rf_model = None

def load_rf(path: str = "models/rf_model.pkl"):
    """
    Load the Random Forest model from disk (only once).
    """
    global _rf_model
    if _rf_model is None:
        _rf_model = joblib.load(path)
    return _rf_model

def predict_from_pil_rf(pil_img: Image.Image):
    """
    Predict a digit from a PIL image using the Random Forest.
    Returns (prediction:int, probabilities:list).
    """
    model = load_rf()
    flat, _ = preprocess_pil_for_models(pil_img)
    probs = model.predict_proba(flat)[0]
    pred = int(np.argmax(probs))
    return pred, probs.tolist()

def predict_digit_rf(img: np.ndarray):
    """
    Streamlit entrypoint.
    Takes a (28,28) numpy array from the canvas,
    converts to PIL, then calls predict_from_pil_rf.
    """
    pil_img = Image.fromarray((img * 255).astype("uint8"))
    return predict_from_pil_rf(pil_img)
