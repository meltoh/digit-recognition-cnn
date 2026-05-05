
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist


MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "digit_cnn_model.h5")


def load_and_prepare_data():
    """
    Loads the MNIST dataset and prepares it for CNN training.
    MNIST contains 28x28 grayscale images of handwritten digits from 0 to 9.
    """

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize pixel values from 0-255 to 0-1
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Reshape images from (28, 28) to (28, 28, 1)
    # The final 1 means grayscale image channel
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, y_train, x_test, y_test


def build_cnn_model():
    """
    Builds a simple CNN model for handwritten digit recognition.
    """

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),

        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def train_and_save_model():
    """
    Trains the CNN model and saves it inside the model folder.
    """

    os.makedirs(MODEL_DIR, exist_ok=True)

    x_train, y_train, x_test, y_test = load_and_prepare_data()

    model = build_cnn_model()

    print("Training model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=64,
        validation_split=0.1
    )

    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test)

    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    model.save(MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save_model()