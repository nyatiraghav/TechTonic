import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import json

# Load the class indices
with open('class_indices.json', 'r') as f:
    class_indices = json.load(f)

# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image):
    # Resize the image
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    img_array = np.array(image)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image):
    preprocessed_img = load_and_preprocess_image(image)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Load the trained model
model = load_model('potato.h5')

# Streamlit App
st.title('Potato Disease Classification')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    predicted_class_name = predict_image_class(model, image)
    st.write("Prediction:", predicted_class_name)
