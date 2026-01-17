from flask import Flask, render_template, request, jsonify
import os as os
from Model_test import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
print("Model_test.py loaded successfully")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/classify", methods=["POST"])
def classify():
    image = request.files["image"]
    model = request.form["model"]

    image_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(image_path)

    prediction, accuracy = predict_image(image_path, model)
    
    return jsonify({
        "image_url": image_path,
        "prediction": prediction,
        "accuracy": accuracy
    })

def predict(image, model):
    return f"Prediction using {model}: CAT"   


if __name__ =='__main__':
    app.run(port=8000,debug=True)


