"""CNN architecture for binary plant-disease classification (healthy vs diseased)."""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()
