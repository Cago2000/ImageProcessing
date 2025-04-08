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
            counter += relation_function(image, y, x)
    return counter
