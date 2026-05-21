from models import loader
from services.preprocessing import array_to_base64, decode_image

def run_predict(b64_str: str) -> dict:
    tensor = decode_image(b64_str)
    conv1_out, conv2_out, preds = loader.activation_model.predict(tensor)
    return {
        "predictions": preds[0].tolist(),
        "feature_maps": {
            "conv2d_1": [array_to_base64(conv1_out[0, :, :, i]) for i in range(conv1_out.shape[-1])],
            "conv2d_2": [array_to_base64(conv2_out[0, :, :, i]) for i in range(conv2_out.shape[-1])],
        }
    }