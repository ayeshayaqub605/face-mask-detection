import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("😷 Face Mask Detection App")

st.write("Choose input method:")

option = st.radio("Select Option:", ["Upload Image", "Use Camera"])

# Dummy prediction function (replace with your model)
def predict(image):
    # yahan tum apna trained model use kar sakti ho
    return "Mask 😷"  # ya "No Mask ❌"

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
