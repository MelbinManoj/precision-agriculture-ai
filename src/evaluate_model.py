"""Evaluate the trained model on a held-out test set."""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def evaluate_model(model_path, test_dir):
    model = tf.keras.models.load_model(model_path)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        color_mode="grayscale",
        class_mode="binary",
    )

    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")


if __name__ == "__main__":
    model_path = "plant_disease_model.h5"
    test_dir = "path/to/test/data"
    evaluate_model(model_path, test_dir)
