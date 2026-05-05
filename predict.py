import numpy as np
import tensorflow as tf

from utils.preprocessing import preprocess_image
from utils.labels import DIGIT_WORDS


MODEL_PATH = "model/digit_cnn_model.h5"


# Load the trained CNN model once
model = tf.keras.models.load_model(MODEL_PATH)


def predict_digit(image):
    """
    Predicts a handwritten digit from an input image.

    Returns:
    - predicted digit
    - digit in words
    - confidence percentage
    - top 3 predictions
    """

    # Convert image into CNN-ready format
    processed_image = preprocess_image(image)

    # Get prediction probabilities from the CNN model
    predictions = model.predict(processed_image)[0]

    # Get the digit with the highest probability
    predicted_digit = int(np.argmax(predictions))

    # Get confidence score
    confidence = float(predictions[predicted_digit] * 100)

    # Get top 3 predictions
    top_3_indices = predictions.argsort()[-3:][::-1]

    top_3_predictions = []
    for index in top_3_indices:
        top_3_predictions.append({
            "digit": int(index),
            "word": DIGIT_WORDS[int(index)],
            "confidence": float(predictions[index] * 100)
        })

    return {
        "digit": predicted_digit,
        "word": DIGIT_WORDS[predicted_digit],
        "confidence": confidence,
        "top_3": top_3_predictions
    }