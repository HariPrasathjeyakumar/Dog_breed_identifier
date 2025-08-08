import streamlit as st
import tensorflow as tf
import json
import numpy as np
from PIL import Image

# -------------------
# Load Model & Labels
# -------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("dog_breed_classifier.keras")
    return model

@st.cache_resource
def load_labels():
    with open("class_labels.json", "r") as f:
        labels = json.load(f)
    return labels

model = load_model()
label_map = load_labels()

# -------------------
# Helper Functions
# -------------------
IMG_SIZE = 160

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def clean_breed_name(name):
    # Replace underscores with spaces and capitalize words
    return " ".join([word.capitalize() for word in name.replace("_", " ").split()])

# -------------------
# Streamlit UI
# -------------------
st.title("🐶 Dog Breed Classifier")
st.write("Upload a dog image to predict its breed!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_index = np.argmax(predictions)
    predicted_breed = clean_breed_name(label_map[str(predicted_index)])
    confidence = np.max(predictions) * 100

    st.subheader("Prediction Results")
    st.write(f"**Predicted Breed:** {predicted_breed}")
    st.write(f"**Confidence:** {confidence:.2f}%")
