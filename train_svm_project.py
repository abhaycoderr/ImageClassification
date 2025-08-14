import os
import cv2
import numpy as np
from tqdm import tqdm
from skimage.feature import hog
from skimage.transform import resize
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# --- 1. Define Constants and Paths ---
DATA_DIR = 'data/PetImages/'
IMG_SIZE = 128
HOG_CELLS = (8, 8)
HOG_BLOCKS = (2, 2)
HOG_ORIENTATIONS = 9

# --- 2. Load and Preprocess Data Function ---
def load_data(data_dir):
    data = []
    categories = ['Cat', 'Dog']
    print(f"Loading images from {data_dir}...")
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)  # 0 for Cat, 1 for Dog

        for img_file in tqdm(os.listdir(path)):
            try:
                img_path = os.path.join(path, img_file)
                
                if not img_file.endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img_array is None:
                    # Handle files that cannot be read by OpenCV (often corrupted)
                    print(f"Warning: Skipping corrupted image at {img_path}")
                    continue

                resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                data.append([resized_array, class_num])
            except Exception as e:
                print(f"Error processing {img_file}: {e}")
    return np.array(data, dtype=object)

# --- 3. Extract HOG Features Function ---
def extract_hog_features(images):
    hog_features = []
    print("Extracting HOG features...")
    
    for image in tqdm(images):
        resized_image = resize(image, (64, 128))
        fd = hog(resized_image, orientations=HOG_ORIENTATIONS, pixels_per_cell=HOG_CELLS,
                 cells_per_block=HOG_BLOCKS)
        hog_features.append(fd)
    return np.array(hog_features)

# --- 4. Main Script Execution ---
if __name__ == '__main__':
    # Load all image data
    all_data = load_data(DATA_DIR)
    
    # Separate features and labels
    X = np.array([item[0] for item in all_data])
    y = np.array([item[1] for item in all_data])

    # Extract HOG features from all images
    X_hog = extract_hog_features(X)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_hog, y, test_size=0.2, random_state=42)

    # --- 5. Train the SVM Model ---
    print("\nTraining the SVM model...")
    svm_model = SVC(kernel='rbf', C=1.0, gamma='auto')
    svm_model.fit(X_train, y_train)

    # Save the trained model for future use
    joblib.dump(svm_model, 'svm_model.joblib')
    print("Model trained and saved as 'svm_model.joblib'.")

    # --- 6. Evaluate the Model ---
    print("\n--- Model Evaluation ---")
    y_pred = svm_model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Cat', 'Dog']))