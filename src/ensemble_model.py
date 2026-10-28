"""Average the predictions of several trained models for a more reliable result."""
import numpy as np
from tensorflow.keras.models import load_model

import preprocess


def ensemble_predict(image_path, model_paths):
    models = [load_model(model_path) for model_path in model_paths]
    preprocessed_image = preprocess.preprocess_image(image_path)

    if preprocessed_image is None:
        print("Error in preprocessing image")
        return

    predictions = [m.predict(preprocessed_image) for m in models]
    final_prediction = np.mean(predictions, axis=0)

    label = "Healthy" if final_prediction[0][0] < 0.5 else "Diseased"
    print(f"Ensemble Prediction: {label}")
    return label


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    model_paths = ["model1.h5", "model2.h5", "model3.h5"]
    ensemble_predict(image_path, model_paths)
