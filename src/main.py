"""Train the CNN on a folder-structured dataset and save it as plant_disease_model.h5.

Expected layout:
    data/train/healthy/*.jpg   data/train/diseased/*.jpg
    data/val/healthy/*.jpg     data/val/diseased/*.jpg
"""
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import model as cnn

EPOCHS = 50


def train_model(train_dir, val_dir):
    model = cnn.create_model()

    train_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        color_mode="grayscale",
        class_mode="binary",
    )

    val_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=32,
        color_mode="grayscale",
        class_mode="binary",
    )

    model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=val_generator,
        validation_steps=val_generator.samples // val_generator.batch_size,
        epochs=EPOCHS,
    )

    model.save("plant_disease_model.h5")


if __name__ == "__main__":
    train_dir = "path/to/training/data"
    val_dir = "path/to/validation/data"
    train_model(train_dir, val_dir)
