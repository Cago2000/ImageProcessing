import cv2
import numpy as np
import bounding_box
import colors


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

def draw_bounding_boxes(image: np.ndarray, blobs: list, min_box_area: int, box_color: list[int]) -> tuple:
    result = image.copy()
    bounding_boxes = []
    for blob in blobs:
        ys, xs = zip(*blob)
        top, left = min(ys), min(xs)
        bottom, right = max(ys), max(xs)

        width = right - left + 1
        height = bottom - top + 1
        area = width * height

        if area < min_box_area:
            continue

        result[top, left:right+1] = box_color
        result[bottom, left:right+1] = box_color

        result[top:bottom+1, left] = box_color
        result[top:bottom+1, right] = box_color

        bounding_box_obj = bounding_box.BoundingBox(((top + bottom) // 2), ((right + left) // 2), height, width, area, colors.get_str_from_color(box_color))
        bounding_boxes.append(bounding_box_obj)
    return result, bounding_boxes

'''
(178, 178) 11 189 189 11
(50, 13) 75 125 123 110
(22, 12) 75 97 166 154
'''


