import math

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

def mean(image: np.ndarray) -> np.float64:
    flattened_image = image.flatten()
    return np.sum(flattened_image) / len(flattened_image)

def variance(image: np.ndarray) -> np.float64:
    mean_value = mean(image)
    flattened_image = image.flatten()
    return np.sum((flattened_image**2 - mean_value**2)) / len(flattened_image)

def std(image: np.ndarray) -> np.float64:
    return np.float64(math.sqrt(variance(image)))

def histogram(image: np.ndarray) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    flattened_image = image.flatten()
    max_value = np.max(flattened_image)
    image_histogram = np.zeros((max_value + 1))
    for pixel in flattened_image:
        image_histogram[pixel] += 1
    return image_histogram