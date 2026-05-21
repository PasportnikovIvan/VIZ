import base64
import io
import numpy as np
import tensorflow as tf
from PIL import Image

def decode_image(b64_str: str) -> tf.Tensor:
    """
    Processes a base64-encoded image drawn by the user and converts it into a tensor suitable for model input.
    Args:
        b64_str (str): The base64-encoded image string.
    Returns:
        tf.Tensor: A tensor of shape (1, 28, 28, 1) ready for model input.
    """
    if b64_str.startswith("data:image"):
        b64_str = b64_str.split(",")[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert('L')
    arr = np.array(img.resize((28, 28))).astype('float32') / 255.0
    return tf.convert_to_tensor(arr.reshape(1, 28, 28, 1))

def array_to_base64(arr: np.ndarray) -> str:
    """
    Normalizes a 2D numpy array to [0, 255] and encodes it as a base64 PNG image.
    Args:
        arr (np.ndarray): The 2D feature map array.
    Returns:
        str: A base64-encoded PNG data URI.
    """
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max > arr_min:
        arr_norm = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr_norm = np.zeros_like(arr)
    
    img_uint8 = (arr_norm * 255).astype(np.uint8)
    img = Image.fromarray(img_uint8)
    
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')