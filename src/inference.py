"""Run the trained model on a single image and print / return the prediction."""
import tensorflow as tf

import preprocess


def predict(image_path, model_path):
    model = tf.keras.models.load_model(model_path)
    preprocessed_image = preprocess.preprocess_image(image_path)

    if preprocessed_image is None:
        print("Error in preprocessing image")
        return "Error in preprocessing image"

    prediction = model.predict(preprocessed_image)
    label = "Healthy" if prediction[0][0] < 0.5 else "Diseased"
    print(f"Prediction: {label}")
    return label


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    model_path = "path/to/your/model.h5"
    predict(image_path, model_path)
