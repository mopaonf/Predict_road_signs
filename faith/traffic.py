import cv2
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler

EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Normalize the dataset
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False
    )
    datagen.fit(x_train)

    # Learning rate scheduler
    def lr_schedule(epoch):
        initial_lr = 0.001
        drop = 0.5
        epochs_drop = 5
        return initial_lr * (drop ** (epoch // epochs_drop))

    lr_scheduler = LearningRateScheduler(lr_schedule)

    # Get a compiled neural network
    model = get_model()

    # Fit model on augmented training data
    model.fit(
        datagen.flow(x_train, y_train, batch_size=32),
        epochs=EPOCHS,
        validation_data=(x_test, y_test),
        callbacks=[lr_scheduler]
    )

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
    else:
        filename = "best_model.h5"  # Default filename
    model.save(filename)
    print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.
    """
    images = []
    labels = []

    # Iterate through each category directory
    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))
        if not os.path.isdir(category_path):
            continue

        # Iterate through each image in the category directory
        for image_file in os.listdir(category_path):
            image_path = os.path.join(category_path, image_file)
            # Read and resize the image
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
                images.append(image)
                labels.append(category)

    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model.
    """
    model = tf.keras.models.Sequential([
        # Convolutional layer with 64 filters
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Second convolutional layer with 128 filters
        tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Third convolutional layer with 256 filters
        tf.keras.layers.Conv2D(256, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Flatten the output
        tf.keras.layers.Flatten(),

        # Add a dense hidden layer with 512 units
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Output layer with NUM_CATEGORIES units
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
