# app.py
%%writefile app.py
import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

# load model
model = tf.keras.models.load_model("mobilenet_mask_model.keras")
st.title("😷 Face Mask Detection App")

st.write("Choose input method:")

option = st.radio("Select Option:", ["Upload Image", "Use Camera"])

# Dummy prediction function (replace with your model)
def predict(image):
    img = cv2.resize(image, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    class_id = np.argmax(pred)

    if class_id == 0:
        return "Mask 😷"
    else:
        return "No Mask ❌"
# =======================
# 📁 Upload Image
# =======================
if option == "Upload Image":
    file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if file is not None:
        image = Image.open(file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = np.array(image)

        result = predict(img)
        st.success(f"Prediction: {result}")

# =======================
# 📷 Camera Input
# =======================
elif option == "Use Camera":
    img_file_buffer = st.camera_input("Take a picture")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        st.image(image, caption="Captured Image", use_column_width=True)

        img = np.array(image)

        result = predict(img)
        st.success(f"Prediction: {result}")
