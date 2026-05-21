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

def run_get_kernel_weights(layer_name: str, kernel_idx: int) -> dict:
    try:
        layer = loader.activation_model.get_layer(layer_name)
        weights = layer.get_weights()[0] # Shape: (3, 3, in_channels, out_channels)
    except Exception as e:
        return {"error": f"Layer lookup failed: {str(e)}"}

    kernel_slice = weights[:, :, :, kernel_idx]
    in_channels = kernel_slice.shape[2]

    decomposed_grids = []
    for c in range(in_channels):
        grid = kernel_slice[:, :, c]
        g_min, g_max = grid.min(), grid.max()
        if (g_max - g_min) > 0:
            grid_norm = (grid - g_min) / (g_max - g_min)
        else:
            grid_norm = grid
        decomposed_grids.append(grid_norm.tolist())

    return {
        "layer_name": layer_name,
        "kernel_idx": kernel_idx,
        "decomposed_weights": decomposed_grids
    }