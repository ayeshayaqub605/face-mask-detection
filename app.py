import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model (cache for fast loading)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mobilenet_mask_model.keras")

model = load_model()

st.title("😷 Face Mask Detection App")

option = st.radio("Select Option:", ["Upload Image", "Use Camera"])

def predict(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # fix color issue
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)

    return "Mask 😷" if class_id == 0 else "No Mask ❌"

# Upload
if option == "Upload Image":
    file = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        st.image(image, use_column_width=True)

        result = predict(np.array(image))
        st.success(result)

# Camera
elif option == "Use Camera":
    cam = st.camera_input("Take picture")

    if cam:
        image = Image.open(cam)
        st.image(image, use_column_width=True)

        result = predict(np.array(image))
        st.success(result)
