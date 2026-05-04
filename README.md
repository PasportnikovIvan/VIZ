# CNN MNIST Visual Analytics Dashboard

An interactive Visual Analytics Dashboard designed to demystify the inner workings of a Convolutional Neural Network (CNN) trained on the MNIST dataset. This tool allows users to peek inside the "black box" of a neural network by visualizing its learning process over time, exploring its 10-dimensional output manifold, and inspecting real-time intermediate convolutional feature maps.

## ✨ Features

- **Global View (Manifold Exploration)**: An animated D3.js scatter plot tracking the model's predictions across multiple training epochs (0, 1, 3, 5, 10, 15). Switch seamlessly between **t-SNE**, **PCA**, and custom **Star Coordinates** projections to see how the network learns to cluster digits.
- **Diagnostic View (Dynamic Confusion Matrix)**: An interactive confusion matrix that updates based on the selected training epoch. Click on any cell to filter the Global View and isolate specific correct predictions or misclassifications.
- **Local View (Live Inference Engine)**: Draw a digit on the canvas, and the FastAPI backend will instantly run inference. The dashboard visualizes the prediction confidence distribution alongside the actual **32 and 64 feature maps** extracted from the network's hidden Conv2D layers.
- **Responsive UI & Theming**: A premium, glassmorphism-inspired UI featuring a seamless Light/Dark mode toggle and a fully responsive layout for desktop and tablet screens.

## 🛠️ Technology Stack

- **Machine Learning**: Python, TensorFlow / Keras (Functional API), scikit-learn, openTSNE.
- **Backend**: FastAPI, Uvicorn, Pillow, NumPy.
- **Frontend**: HTML5, CSS Variables, Vanilla JavaScript, D3.js (v7).

## 🚀 Installation & Setup

### 1. Clone the repository and set up a virtual environment
```bash
# Create a virtual environment (Windows)
python -m venv venv
.\venv\Scripts\activate

# Or on macOS/Linux:
# python3 -m venv venv
# source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate the ML Models and Data
Before starting the server, you need to train the CNN, save its epoch checkpoints, and generate the dimensionality reduction projections (`data.json`).
```bash
python train_and_extract.py
```
*(Note: This process trains the model for 15 epochs and calculates t-SNE, PCA, and Star Coordinates for 1000 test images. It may take a few minutes depending on your hardware).*

### 4. Start the Application Server
Once `data.json` and `mnist_cnn.keras` are generated, start the FastAPI backend:
```bash
uvicorn app:app --reload
```

### 5. Access the Dashboard
Open your web browser and navigate to:
**http://127.0.0.1:8000**

## 📂 Project Structure

- `app.py`: FastAPI application. Serves the static files and hosts the `/predict` endpoint, which intercepts and returns intermediate layer activations using a sliced Keras functional model.
- `train_and_extract.py`: The machine learning pipeline. Builds the CNN, trains it on MNIST, saves weights at designated epochs, runs dimensionality reduction (t-SNE/PCA/Star), and outputs `data.json`.
- `verify_data.py`: A quick console script to verify model accuracy and calculate t-SNE centroids.
- `static/`
  - `index.html`: The main dashboard layout.
  - `styles.css`: CSS styles, CSS variables, Light/Dark mode logic, and responsive media queries.
  - `main.js`: The frontend logic. Handles fetching `data.json`, D3.js scatter plot animations, confusion matrix interactions, canvas drawing, and inference requests.
