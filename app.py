import os
import io
import base64
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image

# Ensure static directory exists so FastAPI doesn't fail on startup
os.makedirs("static", exist_ok=True)

app = FastAPI(title="MNIST CNN Visual Analytics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
main_model = None
activation_model = None

@app.on_event("startup")
async def load_model():
    """Initializes and loads the native Keras model and extracts the intermediate activation sub-model."""
    global main_model, activation_model
    try:
        main_model = tf.keras.models.load_model('mnist_cnn.keras')
        
        layer_outputs = [
            main_model.layers[1].output, # Conv2D 1 (32 filters)
            main_model.layers[3].output, # Conv2D 2 (64 filters)
            main_model.layers[-1].output # Dense (Predictions)
        ]
        activation_model = tf.keras.models.Model(inputs=main_model.input, outputs=layer_outputs)
    except Exception as e:
        print(f"Error loading model on startup: {e}")

class PredictRequest(BaseModel):
    image: str

def array_to_base64(arr):
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

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Processes a base64-encoded image drawn by the user and executes inference.
    Returns the network's confidence distribution and intermediate Conv2D feature maps.
    """
    if activation_model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded on the server.")
        
    try:
        b64_str = request.image
        if b64_str.startswith("data:image"):
            b64_str = b64_str.split(",")[1]
            
        img_bytes = base64.b64decode(b64_str)
        img = Image.open(io.BytesIO(img_bytes)).convert('L')
        img = img.resize((28, 28))
        
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)
        
        conv1_out, conv2_out, preds = activation_model.predict(img_array)
        
        feature_maps = {
            "conv2d_1": [],
            "conv2d_2": []
        }
        
        for i in range(conv1_out.shape[-1]):
            fmap = conv1_out[0, :, :, i]
            feature_maps["conv2d_1"].append(array_to_base64(fmap))
            
        for i in range(conv2_out.shape[-1]):
            fmap = conv2_out[0, :, :, i]
            feature_maps["conv2d_2"].append(array_to_base64(fmap))
            
        return {
            "predictions": preds[0].tolist(),
            "feature_maps": feature_maps
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_dummy():
    """Silences 404 errors emitted by Chrome DevTools."""
    return {}

@app.get("/data.json")
async def get_data_json():
    """
    Serves the pre-computed manifold projections and model history dataset.
    """
    if os.path.exists("data.json"):
        return FileResponse("data.json")
    else:
        raise HTTPException(status_code=404, detail="data.json not found. Run train_and_extract.py first.")

# Mount the static directory to serve index.html, styles.css, main.js
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
