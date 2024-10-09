import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Streamlit app
st.title("Waste Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load and preprocess the image
    image = Image.open(uploaded_file).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Prepare the image for prediction
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Predict the class
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Display the prediction
    st.write(f"Class: {class_name[2:]}")
    st.write(f"Confidence Score: {confidence_score:.2f}")
