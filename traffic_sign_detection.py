from typing import Callable

import cv2
import numpy as np


def match_template(image, template, threshold=0.8):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    boxes = []
    for pt in zip(*locations[::-1]):
        boxes.append((pt, (pt[0] + template.shape[1], pt[1] + template.shape[0])))

    return boxes



def get_blobs(mask: np.ndarray) -> list:
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

def draw_bounding_boxes(image:np.ndarray, blobs: list, min_box_area: int) -> tuple:
    result = image.copy()
    center_positions = []
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
        center_position = ((top + bottom) // 2), ((right + left) // 2), area
        center_positions.append(center_position)
    return result, center_positions

'''
(178, 178) 11 189 189 11
(50, 13) 75 125 123 110
(22, 12) 75 97 166 154
'''


