"""Live camera feed for monitoring the greenhouse. Press 'q' to quit."""
import cv2


def capture_and_monitor():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read frame")
            break

        cv2.imshow("Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_and_monitor()
