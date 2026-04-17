import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("mobilenet_mask_model.keras")

st.title("Face Mask Detection App 😷")

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file is not None:
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    st.image(img, channels="BGR")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        st.warning("No face detected 😐")
    else:
        for (x, y, w, h) in faces:
            face = img[y:y+h, x:x+w]

            face = cv2.resize(face, (128,128))
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = model.predict(face)

            # DEBUG
            st.write("Prediction:", pred)

            # Softmax model (tumhara)
            if np.argmax(pred) == 0:
                st.success("Mask Detected 😷")
            else:
                st.error("No Mask 😐")
