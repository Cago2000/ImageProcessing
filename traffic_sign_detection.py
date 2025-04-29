import cv2
import numpy as np


def match_template(image, template, threshold=0.8):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    boxes = []
    for pt in zip(*locations[::-1]):
        boxes.append((pt, (pt[0] + template.shape[1], pt[1] + template.shape[0])))

    return boxes

def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[2], rgb[1], rgb[0]
    r, g, b = r / 255, g / 255, b / 255
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta == 0:
        h = 0
    else:
        match max_c:
            case x if x == r:
                h = 60 * (((g - b) / delta) % 6)
            case x if x == g:
                h = 60 * (((b - r) / delta) + 2)
            case x if x == b:
                h = 60 * (((r - g) / delta) + 4)
    s = 0 if max_c == 0 else delta / max_c
    v = max_c
    return np.array([h, s, v])

def is_strong_red(h: int, s: np.float64, v: np.float64) -> bool:
    is_hue_red = h >= 345 or h <= 15
    is_saturated = s >= 0.5
    is_bright_enough = v >= 0.2
    return is_hue_red and is_saturated and is_bright_enough

def label_connected_components(mask):
    label = 1
    labels = np.zeros_like(mask, dtype=np.int32)
    height, width = mask.shape

    def flood_fill(y, x, label):
        stack = [(y, x)]
        coords = []

        while stack:
            cy, cx = stack.pop()
            if 0 <= cy < height and 0 <= cx < width and mask[cy, cx] == 1 and labels[cy, cx] == 0:
                labels[cy, cx] = label
                coords.append((cy, cx))
                # 4-connectivity
                stack.extend([(cy+1, cx), (cy-1, cx), (cy, cx+1), (cy, cx-1)])
        return coords

    blobs = []
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 1 and labels[y, x] == 0:
                coords = flood_fill(y, x, label)
                blobs.append(coords)
                label += 1
    return blobs

def draw_bounding_boxes(image, blobs, min_box_area=50):
    result = image.copy()
    for blob in blobs:
        ys, xs = zip(*blob)
        top, left = min(ys), min(xs)
        bottom, right = max(ys), max(xs)

        width = right - left + 1
        height = bottom - top + 1
        area = width * height

        if area < min_box_area:
            continue

        result[top, left:right+1] = [0, 255, 0]     # Red
        result[bottom, left:right+1] = [0, 255, 0]  # Red

        result[top:bottom+1, left] = [0, 255, 0]
        result[top:bottom+1, right] = [0, 255, 0]
    return result


def get_red_mask(image):
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            hsv = rgb_to_hsv(image[y, x])
            h, s, v = hsv[2], hsv[1], hsv[0]
            if is_strong_red(h, s, v):
                mask[y, x] = 1
    return mask

