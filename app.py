import streamlit as st
from model_script import predict_caption, load_model_and_tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os
from PIL import Image

# Load model and tokenizer
model, tokenizer, max_length, features = load_model_and_tokenizer()

st.title("Image Caption Generator")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_path = "temp.jpg"
    image.save(img_path)
    
    # Process image
    img = load_img(img_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    img = preprocess_input(img)
    
    # Extract features
    feature = features.predict(img, verbose=0)
    
    # Generate caption
    caption = predict_caption(model, feature, tokenizer, max_length)
    
    st.subheader("Generated Caption:")
    st.write(caption)