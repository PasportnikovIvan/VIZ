import os
import json
import base64
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from openTSNE import TSNE
import io
from PIL import Image

def build_model():
    """
    Constructs and compiles the MNIST CNN architecture using the Keras Functional API.
    
    Returns:
        tf.keras.models.Model: A compiled Keras model.
    """
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="conv2d_1")(inputs)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="conv2d_2")(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(10, activation="softmax")(x)
    
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

class EpochModelCheckpoint(callbacks.Callback):
    """
    Custom Keras callback to save model weights at specific designated epochs.
    """
    def __init__(self, save_epochs):
        super().__init__()
        self.save_epochs = save_epochs
        
    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        if current_epoch in self.save_epochs:
            self.model.save_weights(f'weights_epoch_{current_epoch}.weights.h5')

def image_to_base64(img_array):
    """
    Converts a normalized 2D image array to a base64 encoded PNG string.
    
    Args:
        img_array (np.ndarray): The image array, scaled between 0 and 1.
        
    Returns:
        str: A base64 PNG data URI.
    """
    img_uint8 = (img_array * 255).astype(np.uint8)
    if len(img_uint8.shape) == 3 and img_uint8.shape[-1] == 1:
        img_uint8 = img_uint8.squeeze()
    img = Image.fromarray(img_uint8)
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_star_coordinates(predictions):
    """
    Projects 10-dimensional network predictions into a 2D space using Star Coordinates.
    
    Args:
        predictions (np.ndarray): An (N, 10) array of softmax output probabilities.
        
    Returns:
        np.ndarray: An (N, 2) array of 2D coordinates.
    """
    num_dims = 10
    angles = np.linspace(0, 2 * np.pi, num_dims, endpoint=False)
    axes_x = np.cos(angles)
    axes_y = np.sin(angles)
    
    x = np.dot(predictions, axes_x)
    y = np.dot(predictions, axes_y)
    return np.column_stack((x, y))

class PCA():
    """Custom implementation of PCA using numpys svd"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):

        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        self.components = Vt[:self.n_components]
        return self

    def transform(self, X):
        X_centered = X - self.mean
        
        return np.dot(X_centered, self.components.T)

def main():
    """
    Executes the complete machine learning pipeline: training the CNN, saving epoch weights,
    performing dimensionality reduction (t-SNE, PCA, Star), and exporting data.json.
    """
    # Load Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
    
    # Subset test data to 1000 images
    subset_size = 1000
    x_test_subset = x_test[:subset_size]
    y_test_subset = y_test[:subset_size]

    # Build and Train Model
    model = build_model()
    save_epochs = [0, 1, 3, 5, 10, 15]
    total_epochs = 15
    
    print("Saving Epoch 0 random weights...")
    model.save_weights('weights_epoch_0.weights.h5')
    
    print("Training model...")
    model.fit(
        x_train, y_train,
        epochs=total_epochs,
        batch_size=128,
        validation_data=(x_test_subset, y_test_subset),
        callbacks=[EpochModelCheckpoint(save_epochs)]
    )
    
    # Save the final model (full architecture + weights)
    model.save('mnist_cnn.keras')
    
    # Extract Predictions
    print("\nExtracting predictions for tracked epochs...")
    all_predictions = {}
    for epoch in save_epochs:
        print(f"Loading weights for epoch {epoch}...")
        model.load_weights(f'weights_epoch_{epoch}.weights.h5')
        preds = model.predict(x_test_subset)
        all_predictions[epoch] = preds

    # Dimensionality Reduction setup on final epoch
    print("\nFitting PCA and t-SNE on final epoch (Epoch 15)...")
    final_preds = all_predictions[15]
    
    pca = PCA(n_components=2)
    pca.fit(final_preds)
    
    tsne = TSNE(
        n_components=2,
        perplexity=30,
        metric="euclidean",
        n_jobs=-1,
        random_state=42
    )
    tsne_embedding = tsne.fit(final_preds)

    # Transform all epochs and structure JSON
    print("\nApplying transformations and generating JSON structure...")
    
    # Pre-encode images to base64
    b64_images = [image_to_base64(img) for img in x_test_subset]
    
    output_data = []
    
    # Process each epoch's projections
    epoch_projections = {}
    for epoch in save_epochs:
        print(f"Transforming space for epoch {epoch}...")
        preds = all_predictions[epoch]
        
        # PCA Transform
        pca_proj = pca.transform(preds)
        
        # t-SNE Transform
        if epoch == 15:
            tsne_proj = np.array(tsne_embedding)
        else:
            tsne_proj = tsne_embedding.transform(preds)
            
        # Star Coordinates Transform
        star_proj = calculate_star_coordinates(preds)
        
        predicted_labels = np.argmax(preds, axis=1)
        
        epoch_projections[epoch] = {
            'predicted_label': predicted_labels.tolist(),
            'pca': pca_proj.tolist(),
            'tsne': tsne_proj.tolist(),
            'star': star_proj.tolist()
        }

    # Restructure into array of image objects
    print("\nStructuring final JSON file...")
    for i in range(subset_size):
        item = {
            'id': i,
            'true_label': int(y_test_subset[i]),
            'image_b64': b64_images[i],
            'epochs': {}
        }
        for epoch in save_epochs:
            item['epochs'][str(epoch)] = {
                'predicted_label': int(epoch_projections[epoch]['predicted_label'][i]),
                'pca': epoch_projections[epoch]['pca'][i],
                'tsne': epoch_projections[epoch]['tsne'][i],
                'star': epoch_projections[epoch]['star'][i]
            }
        output_data.append(item)
        
    with open('data.json', 'w') as f:
        json.dump(output_data, f)
        
    print(f"\nDone! Extracted data for {subset_size} images exported to data.json")
    print("Files created:")
    print(" - data.json")
    print(" - mnist_cnn.keras")
    for epoch in save_epochs:
        print(f" - weights_epoch_{epoch}.weights.h5")

if __name__ == '__main__':
    main()
