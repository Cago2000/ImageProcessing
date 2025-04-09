import numpy as np
from typing import Callable

def co_occurrence(image: np.ndarray, relation_function: Callable[[np.ndarray, int, int], bool]) -> int:
    if len(image.shape) == 3:
        height, width, _ = image.shape
    else:
        height, width = image.shape

    counter = 0
    for y in range(height):
        for x in range(width):
            counter += relation_function(image, x, y)
    return counter

def median(image: np.ndarray) -> np.ndarray:
    return image[image.shape[0] // 2][image.shape[1] // 2]

def mean(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        height, width, _ = image.shape
        mean_value = (0, 0, 0)
    else:
        height, width = image.shape
        mean_value = 0
    for y in range(height):
        for x in range(width):
            mean_value += image[y, x]
    pixel_amount = image.shape[0]*image.shape[1]
    print(pixel_amount)
    mean_value = mean_value/pixel_amount
    return mean_value