import cv2
import numpy as np
import random

DIRT_PALETTE = {
    "mud": [(30, 67, 101), (20, 50, 80)],
    "oil": [(20, 20, 20), (50, 50, 50)],
    "grass": [(34, 139, 34), (50, 205, 50)],
    "wine": [(100, 10, 40), (80, 0, 20)],
    "clean": []
}


def create_random_blob(shape, num_blobs=5):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    for _ in range(num_blobs):
        center_x = random.randint(0, width)
        center_y = random.randint(0, height)
        axes_x = random.randint(10, 80)
        axes_y = random.randint(10, 80)
        angle = random.randint(0, 360)
        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y), angle, 0, 360, 255, -1)

    return mask


def add_dirt(image, dirt_type="mud", max_intensity=0.8):
    if dirt_type == "clean" or dirt_type not in DIRT_PALETTE:
        return image, 0, "clean"

    h, w, _ = image.shape
    dirt_layer = np.zeros_like(image)
    color = random.choice(DIRT_PALETTE[dirt_type])
    dirt_layer[:] = color

    num_splashes = random.randint(3, 15)
    mask = create_random_blob((h, w), num_blobs=num_splashes)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)

    alpha = (mask.astype(float) / 255.0) * max_intensity

    dirty_image = image.copy()
    for c in range(3):
        dirty_image[:, :, c] = (alpha * dirt_layer[:, :, c] + (1 - alpha) * image[:, :, c])

    coverage = np.sum(alpha) / (h * w)
    intensity_score = min(10, int(coverage * 100))

    return dirty_image.astype(np.uint8), intensity_score, dirt_type

