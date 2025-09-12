import tensorflow as tf
import numpy as np
from PIL import Image
from src.utils import preprocess_pil_for_models

# Lazy-loaded singleton CNN model
_cnn_model = None

def load_cnn(path: str = "models/model.h5"):
    """
    Load the CNN model from disk (only once).
    """
    global _cnn_model
    if _cnn_model is None:
        _cnn_model = tf.keras.models.load_model(path)
    return _cnn_model

def predict_from_pil_cnn(pil_img: Image.Image):
    """
    Predict a digit from a PIL image using the CNN.
    Returns (prediction:int, probabilities:list).
    """
    model = load_cnn()
    _, cnn = preprocess_pil_for_models(pil_img)
    probs = model.predict(cnn, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs.tolist()

def predict_digit_cnn(img: np.ndarray):
    """
    Streamlit entrypoint.
    Takes a (28,28) numpy array from the canvas,
    converts to PIL, then calls predict_from_pil_cnn.
    """
    pil_img = Image.fromarray((img * 255).astype("uint8"))
    return predict_from_pil_cnn(pil_img)
