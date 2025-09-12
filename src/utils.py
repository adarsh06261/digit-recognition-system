import io, base64
import numpy as np
from PIL import Image, ImageOps

IMG_SIZE = 28

def dataurl_to_pil(data_url: str) -> Image.Image:
    """Convert a data URL 'data:image/png;base64,...' to grayscale PIL image."""
    header, b64 = data_url.split(",", 1)
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("L")

def preprocess_pil_for_models(pil_img: Image.Image):
    """
    Invert (canvas is black on white), resize to 28x28, normalize to [0,1].
    Returns:
      flat -> (1, 784) for RandomForest
      cnn  -> (1, 28, 28, 1) for Keras
    """
    # Invert to match MNIST (white digit on black)
    pil_img = ImageOps.invert(pil_img)

    # (optional) center-square crop to avoid warping if not square
    w, h = pil_img.size
    if w != h:
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        pil_img = pil_img.crop((left, top, left + side, top + side))

    pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.array(pil_img, dtype="float32") / 255.0
    flat = arr.reshape(1, -1)
    cnn  = arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return flat, cnn
