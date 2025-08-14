import os
import cv2
import numpy as np
from skimage.feature import hog
from skimage.transform import resize
import joblib

def predict_image(image_path, model_path='svm_model.joblib'):
    # Load the trained model
    svm_model = joblib.load(model_path)
    
    # --- Preprocess the new image exactly like the training data ---
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_array is None:
        return "Error: Unable to read image file."

    resized_array = cv2.resize(img_array, (128, 128))
    
    # --- Extract HOG features from the new image ---
    resized_for_hog = resize(resized_array, (64, 128))
    hog_features = hog(resized_for_hog, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2))
    
    # The model expects a 2D array, so we reshape the features
    hog_features = hog_features.reshape(1, -1)
    
    # --- Make the prediction ---
    prediction = svm_model.predict(hog_features)[0]
    
    # Return the predicted class label
    if prediction == 0:
        return "Cat"
    else:
        return "Dog"

# Example usage:
if __name__ == '__main__':
    # Make sure to replace this with the path to a new image
    test_image_path = "data/PetImages/Dog/1.jpg"
    
    if os.path.exists(test_image_path):
        result = predict_image(test_image_path)
        print(f"The model predicts: {result}")
    else:
        print(f"Error: Could not find image at {test_image_path}")