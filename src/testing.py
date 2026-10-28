"""Quick check that images can be read and displayed (camera / dataset sanity test)."""
import cv2


def test_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to open image at {image_path}")
        return

    cv2.imshow("Test Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    image_path = "path/to/your/image.jpg"
    test_image(image_path)
