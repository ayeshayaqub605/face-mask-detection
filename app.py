import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("mobilenet_mask_model.keras")

# Prediction function
def predict(image):
    # Convert PIL image to numpy
    image = image.resize((128, 128))
    img_array = np.array(image)

    # Normalize
    img_array = img_array / 255.0

    # Expand dimensions (batch)
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    pred = model.predict(img_array)[0]

    class_id = np.argmax(pred)
    confidence = float(np.max(pred)) * 100

    if class_id == 0:
        return f"😷 Mask Detected ({confidence:.2f}%)"
    else:
        return f"❌ No Mask Detected ({confidence:.2f}%)"


# UI
demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Face Image"),
    outputs=gr.Label(label="Prediction Result"),
    title="😷 Face Mask Detection App",
    description="Upload a face image and the model will detect whether the person is wearing a mask or not.",
    theme="default"
)

demo.launch()
