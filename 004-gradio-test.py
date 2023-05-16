import gradio as gr
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import pickle
from PIL import Image
import numpy as np
import tensorflow as tf

# Define the GELU layer as before
class GELU(Layer):
    def call(self, x):
        return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

def predict_font_and_style(input_image):
    image_size = (512, 512)

    # Load the trained model and LabelEncoder
    model = load_model('my_model.h5', custom_objects={'GELU': GELU})
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)

    # Convert the input image to a PIL Image object and resize
    image = Image.fromarray((input_image * 255).astype(np.uint8)).resize(image_size)

    # Normalize and expand dimensions
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    # Use the model to make a prediction
    prediction = model.predict(image)

    # Assume prediction includes both font type and style, separated by an underscore
    # Get the index of the highest probability
    predicted_index = np.argmax(prediction)

    # Decode the prediction
    predicted_font_and_style = le.inverse_transform([predicted_index])

    # Check if the string contains an underscore before trying to split it
    if '_' in predicted_font_and_style[0]:
        font_type, font_style = predicted_font_and_style[0].split('_')
        return {"Font Type": font_type, "Font Style": font_style}
    else:
        return {"Font Type": predicted_font_and_style[0], "Font Style": "Unknown"}

iface = gr.Interface(
    fn=predict_font_and_style,
    inputs="image",
    outputs="text",
    title="Font Recognition",
    description="Upload an image and the model will predict the font used in the image."
)
iface.launch(inbrowser=bool)
