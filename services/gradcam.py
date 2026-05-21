import numpy as np
import tensorflow as tf
from PIL import Image
from models import loader
from services.preprocessing import array_to_base64, decode_image
import matplotlib.cm as cm

def run_gradcam(b64_str: str, class_index: int = -1) -> dict:
    tensor = decode_image(b64_str)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = loader.gradcam_model(tensor)
        tape.watch(conv_outputs)
        idx = class_index if class_index != -1 else int(tf.argmax(predictions[0]))
        score = predictions[:, idx]

    grads = tape.gradient(score, conv_outputs)
    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.nn.relu(tf.reduce_sum(weights[0] * conv_outputs[0], axis=-1)).numpy()

    if cam.max() > 0:
        cam = cam / cam.max()

    colormap = cm.get_cmap("turbo")
    heatmap = colormap(cam)[:, :, :3]

    heatmap = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize((28, 28))
    )

    raw_outputs = conv_outputs.numpy()[0]
    raw_grads = grads.numpy()[0]         
    
    channel_contributions = np.mean(raw_outputs * raw_grads, axis=(0, 1))
    most_contributing_idx = int(np.argmax(np.abs(channel_contributions)))

    return {
        "class_index": idx,
        "predicted_label": int(tf.argmax(predictions[0])),
        "heatmap": array_to_base64(heatmap),
        "mostContributingIdx" : most_contributing_idx
    }