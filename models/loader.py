import tensorflow as tf

# Global variables for models
main_model = None
activation_model = None
gradcam_model = None

def load():
    """Initializes and loads the native Keras model and extracts the intermediate activation sub-model."""
    global main_model, activation_model, gradcam_model
    try:
        main_model = tf.keras.models.load_model('mnist_cnn.keras')

        last_conv = main_model.get_layer('conv2d_2')
        activation_model = tf.keras.Model(
            inputs=main_model.input,
            outputs=[
                main_model.get_layer('conv2d_1').output,
                last_conv.output,
                main_model.output
            ]
        )
        gradcam_model = tf.keras.Model(
            inputs=main_model.input,
            outputs=[last_conv.output, main_model.output]
        )
    except Exception as e:
        print(f"Error loading model on startup: {e}")