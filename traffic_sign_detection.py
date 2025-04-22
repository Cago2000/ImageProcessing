import cv2
import numpy as np


def match_template(image, template, threshold=0.8):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)

    boxes = []
    for pt in zip(*locations[::-1]):
        boxes.append((pt, (pt[0] + template.shape[1], pt[1] + template.shape[0])))

    return boxes


