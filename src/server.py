"""Flask web server: upload a leaf image and get a Healthy / Diseased prediction."""
import os

from flask import Flask, render_template, request

import inference

app = Flask(__name__)
UPLOAD_DIR = "uploads"
MODEL_PATH = "plant_disease_model.h5"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file part"

    file = request.files["file"]
    if file.filename == "":
        return "No selected file"

    image_path = os.path.join(UPLOAD_DIR, file.filename)
    file.save(image_path)
    result = inference.predict(image_path, MODEL_PATH)
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
