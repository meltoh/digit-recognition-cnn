import numpy as np
import cv2


def _to_grayscale(image):
    img = np.array(image)

    if img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def _prepare_digit(gray_image):
    """
    Converts a drawn/uploaded digit into MNIST-like format:
    black background, white centered digit, 28x28 size.
    """

    # If image has dark background with light digit, keep it.
    # If image has light background with dark digit, invert it.
    if np.mean(gray_image) > 127:
        gray_image = cv2.bitwise_not(gray_image)

    # Remove light noise and keep digit strokes
    _, thresh = cv2.threshold(gray_image, 30, 255, cv2.THRESH_BINARY)

    coords = cv2.findNonZero(thresh)

    # If canvas is empty, return blank image
    if coords is None:
        blank = np.zeros((28, 28), dtype=np.float32)
        return blank

    x, y, w, h = cv2.boundingRect(coords)
    digit = thresh[y:y + h, x:x + w]

    # Add padding around digit
    pad = 20
    digit = cv2.copyMakeBorder(
        digit,
        pad,
        pad,
        pad,
        pad,
        cv2.BORDER_CONSTANT,
        value=0
    )

    # Resize while preserving aspect ratio
    h, w = digit.shape
    scale = 20.0 / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place digit in center of 28x28 image
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    return canvas.astype("float32") / 255.0


def preprocess_image(image):
    gray = _to_grayscale(image)
    processed = _prepare_digit(gray)
    return processed.reshape(1, 28, 28, 1)


def get_preprocessing_steps(image):
    gray = _to_grayscale(image)

    if np.mean(gray) > 127:
        inverted = cv2.bitwise_not(gray)
    else:
        inverted = gray

    _, thresholded = cv2.threshold(inverted, 30, 255, cv2.THRESH_BINARY)
    final_28 = _prepare_digit(gray)

    return {
        "original": np.array(image),
        "grayscale": gray,
        "thresholded": thresholded,
        "final": final_28
    }

