import numpy as np
import cv2


def _to_grayscale(image):
    img = np.array(image)

    if len(img.shape) == 3 and img.shape[-1] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img


def _center_digit(binary_img):
    coords = cv2.findNonZero(binary_img)

    if coords is None:
        return np.zeros((28, 28), dtype=np.float32)

    x, y, w, h = cv2.boundingRect(coords)
    digit = binary_img[y:y + h, x:x + w]

    h, w = digit.shape
    scale = 20.0 / max(h, w)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = digit

    return canvas.astype("float32") / 255.0


def preprocess_image(image):
    gray = _to_grayscale(image)

    # Resize large uploaded images first for easier processing
    gray = cv2.resize(gray, (280, 280), interpolation=cv2.INTER_AREA)

    # Try both styles:
    # 1. black digit on white background
    # 2. white digit on black background
    inverted = cv2.bitwise_not(gray)

    _, option_1 = cv2.threshold(inverted, 40, 255, cv2.THRESH_BINARY)
    _, option_2 = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY)

    # Choose the version with fewer white pixels.
    # This usually represents the digit, not the background.
    white_pixels_1 = np.sum(option_1 > 0)
    white_pixels_2 = np.sum(option_2 > 0)

    if white_pixels_1 < white_pixels_2:
        final = _center_digit(option_1)
    else:
        final = _center_digit(option_2)

    return final.reshape(1, 28, 28, 1)


def get_preprocessing_steps(image):
    gray = _to_grayscale(image)
    gray_resized = cv2.resize(gray, (280, 280), interpolation=cv2.INTER_AREA)

    inverted = cv2.bitwise_not(gray_resized)

    _, option_1 = cv2.threshold(inverted, 40, 255, cv2.THRESH_BINARY)
    _, option_2 = cv2.threshold(gray_resized, 40, 255, cv2.THRESH_BINARY)

    white_pixels_1 = np.sum(option_1 > 0)
    white_pixels_2 = np.sum(option_2 > 0)

    if white_pixels_1 < white_pixels_2:
        thresholded = option_1
    else:
        thresholded = option_2

    final = _center_digit(thresholded)

    return {
        "original": np.array(image),
        "grayscale": gray_resized,
        "thresholded": thresholded,
        "final": final
    }