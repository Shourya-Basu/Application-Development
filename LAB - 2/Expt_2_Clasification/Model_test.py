import joblib, cv2, numpy as np, tensorflow as tf
import os
from skimage.feature import hog


logistic_model = joblib.load("Models/logistic_model.pkl")
svm_model = joblib.load("Models/svm_model.pkl")
rf_model = joblib.load("Models/rf_model.pkl")
kmeans_data = joblib.load("Models/kmeans_model.pkl")
kmeans_model = kmeans_data["kmeans"]
kmeans_scaler = kmeans_data["scaler"]
cmap = kmeans_data["cluster_map"]
cnn_model = tf.keras.models.load_model("Models/cnn_model.h5")

IMG_SIZE = 128
def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)

    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        transform_sqrt=True
    )
    return features

def preprocess_ml_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    features = extract_hog_features(img)   
    return features.reshape(1, -1)

def preprocess_cnn_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    return img.reshape(1, 128, 128, 3)


def predict_image(image_path, model_name):

    model_name = model_name.upper()   

    if model_name == "CNN":
        X = preprocess_cnn_image(image_path)
        prob = float(cnn_model.predict(X)[0][0])  # P(Dog)

        if prob >= 0.5:
            return "Dog", round(prob * 100, 2)
        else:
            return "Cat", round((1 - prob) * 100, 2)


    if model_name == "KMEAN":  
        X = preprocess_ml_image(image_path)

        # ✅ MUST scale features
        X = kmeans_scaler.transform(X)

        cluster = int(kmeans_model.predict(X)[0])
        mapped = cmap[cluster]   # 0=Cat, 1=Dog

        label = "Dog" if mapped == 1 else "Cat"
        return label, 85.0   
    

    X = preprocess_ml_image(image_path)

    if model_name == "LR":
        pred = int(logistic_model.predict(X)[0])
        acc = float(max(logistic_model.predict_proba(X)[0]) * 100)

    elif model_name == "SVM":
        pred = int(svm_model.predict(X)[0])
        acc = 82.0

    elif model_name == "RF":
        pred = int(rf_model.predict(X)[0])
        acc = float(max(rf_model.predict_proba(X)[0]) * 100)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    label = "Dog" if pred == 1 else "Cat"
    return label, acc
