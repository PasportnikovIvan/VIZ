import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from models import loader
from services import inference, gradcam
from schemas.requests import PredictRequest, GradCAMRequest

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

@app.on_event("startup")
async def startup():
    loader.load()

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Processes a base64-encoded image drawn by the user and executes inference.
    Returns the network's confidence distribution and intermediate Conv2D feature maps.
    """
    if loader.activation_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return inference.run_predict(request.image)

@app.post("/gradcam")
async def gradcam_endpoint(request: GradCAMRequest):
    """
    Generates a Grad-CAM heatmap for the given base64-encoded image and optional class index.
    """
    if loader.gradcam_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    return gradcam.run_gradcam(request.image, request.class_index)

@app.get("/data.json")
async def get_data_json():
    """
    Serves the pre-computed manifold projections and model history dataset.
    """
    if os.path.exists("data.json"):
        return FileResponse("data.json")
    else:
        raise HTTPException(status_code=404, detail="data.json not found. Run train_and_extract.py first.")

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def devtools():
    """Silences 404 errors emitted by Chrome DevTools."""
    return {}

# Mount the static directory to serve index.html, styles.css, main.js
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
