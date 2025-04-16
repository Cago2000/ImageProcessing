import math
import numpy as np
from typing import Callable

from matplotlib import pyplot as plt


def gauss_filter(image: np.ndarray, dim: int) -> np.ndarray | None:
    if len(image.shape) == 3 or dim % 2 == 0:
        return None

    dim = dim // 2
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            pixels = np.array([], dtype=np.uint16)
            for a in range(-dim, dim):
                for b in range(-dim, dim):
                    out_of_bounds = y+a < 0 or y+a >= image.shape[0] or x+b < 0 or x+b >= image.shape[1]
                    if out_of_bounds:
                        continue
                    pixels = np.append(pixels, image[y+a, x+b])
            image[y, x] = np.uint8(np.sum(pixels) // pixels.size)
    return image

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
    flattened_image = image.flatten()
    flattened_image_sorted = sorted(flattened_image)
    return flattened_image_sorted[len(flattened_image_sorted) // 2]

def mean(image: np.ndarray) -> np.float64:
    flattened_image = image.flatten()
    return np.sum(flattened_image) / len(flattened_image)

def variance(image: np.ndarray) -> np.float64:
    mean_value = mean(image)
    flattened_image = image.flatten()
    return np.float64(np.sum((flattened_image - mean_value)**2) / len(flattened_image))

def std(image: np.ndarray) -> np.float64:
    variance_value = variance(image)
    return np.float64(math.sqrt(variance_value))

def entropy(image: np.ndarray) -> np.float64:
    image_histogram = histogram(image)
    entropy_value = np.float64(0.0)
    for gray_value in range(len(image_histogram)):
        probability_of_gray_value = image_histogram[gray_value] / np.sum(image_histogram)
        if probability_of_gray_value > 0:
            entropy_value += probability_of_gray_value * math.log2(probability_of_gray_value)
    return entropy_value * -1

def histogram(image: np.ndarray) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    flattened_image = image.flatten()
    max_value = np.iinfo(image.dtype).max
    image_histogram = np.zeros((max_value+1), dtype=np.uint32)
    for pixel in flattened_image:
        image_histogram[pixel] += 1
    return image_histogram

def relative_histogram(image: np.ndarray) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    image_histogram = histogram(image)
    return image_histogram/image.size

def cumulative_histogram(image: np.ndarray) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    image_histogram = relative_histogram(image)
    max_value = np.iinfo(image.dtype).max
    image_cumulative_histogram = np.zeros((max_value + 1), dtype=np.float64)
    cumulative_sum = 0.0
    for gray_value in range(len(image_histogram)):
        cumulative_sum += image_histogram[gray_value]
        image_cumulative_histogram[gray_value] = cumulative_sum
    return image_cumulative_histogram


def histogram_equalization(image: np.ndarray) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    cumulative_hist = cumulative_histogram(image)

    lookup_table = np.floor(cumulative_hist * 255).astype(np.uint8)

    flattened_image = image.flatten()
    equalized_flattened_image = np.zeros_like(flattened_image, dtype=np.uint8)

    for i in range(flattened_image.size):
        equalized_flattened_image[i] = lookup_table[flattened_image[i]]

    equalized_image = equalized_flattened_image.reshape(image.shape)
    return equalized_image

def gamma_equalization(image: np.ndarray, gamma: float) -> np.ndarray | None:

    max_value = np.iinfo(image.dtype).max
    gray_values = np.linspace(0, max_value, max_value+1)
    lookup_table = np.floor(((gray_values / 255.0)** gamma) * 255).astype(np.uint16)

    flattened_image = image.flatten()
    gamma_equalized_flattened_image = np.zeros_like(flattened_image, dtype=np.uint8)

    for i in range(flattened_image.size):
        gamma_equalized_flattened_image[i] = lookup_table[flattened_image[i]]

    gamma_equalized_image = gamma_equalized_flattened_image.reshape(image.shape)
    return gamma_equalized_image