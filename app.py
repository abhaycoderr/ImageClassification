import streamlit as st
import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
import joblib
from PIL import Image

# --- 1. Load Model and Constants ---
# Load your trained SVM model
svm_model = joblib.load('svm_model.joblib')

IMG_SIZE = 128
HOG_CELLS = (8, 8)
HOG_BLOCKS = (2, 2)
HOG_ORIENTATIONS = 9

# --- 2. Define Prediction Function ---
def predict_image(image_file):
    # Preprocess the new image exactly like the training data
    img = Image.open(image_file).convert('L') # Convert to grayscale
    img_array = np.array(img)
    
    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    
    # Extract HOG features from the new image
    resized_for_hog = resize(resized_array, (64, 128))
    hog_features = hog(resized_for_hog, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2))
    
    # The model expects a 2D array, so we reshape the features
    hog_features = hog_features.reshape(1, -1)
    
    # Make the prediction
    prediction = svm_model.predict(hog_features)[0]
    
    # Return the predicted class label
    if prediction == 0:
        return "Cat"
    else:
        return "Dog"

# --- 3. Streamlit UI ---
st.title("🐱🐶 Dog and Cat Classifier")
st.write("Upload an image of a dog or cat to get a prediction!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Make prediction and display result
    label = predict_image(uploaded_file)
    st.success(f"Prediction: This image is a **{label}**!")