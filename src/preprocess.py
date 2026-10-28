"""Pre-process an image for the CNN: resize -> grayscale -> normalise -> reshape."""
import cv2
import numpy as np

IMG_SIZE = 128


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        return None

    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    # Reshape to (1, 128, 128, 1) for model input
    image = np.reshape(image, (1, IMG_SIZE, IMG_SIZE, 1))
    return image


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    preprocessed_image = preprocess_image(image_path)
    if preprocessed_image is not None:
        print("Image preprocessed successfully")
