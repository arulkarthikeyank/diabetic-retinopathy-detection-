import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("dr_binary_model.h5")

def preprocess_image(uploaded_file):

    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    img = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(img, (224,224))

    img = np.expand_dims(img, axis=0)

    return img

st.title("Diabetic Retinopathy Detection")

uploaded_file = st.file_uploader(
    "Upload Retina Image",
    type=["png","jpg","jpeg"]
)

if uploaded_file is not None:

    image = preprocess_image(uploaded_file)

    prediction = model.predict(image)[0][0]

    st.image(uploaded_file)

    if prediction >= 0.5:

        st.error("Diabetic Retinopathy Detected")

        st.info(f"Confidence: {prediction*100:.2f}%")

    else:

        st.success("No Diabetic Retinopathy")

        st.info(f"Confidence: {(1-prediction)*100:.2f}%")