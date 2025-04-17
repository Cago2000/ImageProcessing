import numpy as np
import statistical_operations as stat_ops

def gray_scale_filter(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    height, width, _ = image.shape
    gray_scaled_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            gray_scaled_image[y][x] = int(0.299 * image[y][x][2] + 0.587 * image[y][x][1] + 0.114 * image[y][x][0])
    return gray_scaled_image

def black_white_filter(image: np.ndarray, threshold: int) -> np.ndarray:
    avg_intensity = np.mean(image, axis=-1)
    output = np.where(avg_intensity >= threshold, 255, 0).astype(np.uint8)
    return output

def blur_filter(image: np.ndarray, kernel_dim: int, kernel_intensity: int) -> np.ndarray:
    kernel = np.ones((kernel_dim, kernel_dim, 3)) / kernel_intensity
    pad_size = kernel.shape[0] // 2
    image = np.pad(
        image,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='constant'
    )
    height, width, channels = image.shape
    k_height, k_width, k_channels = kernel.shape
    out_height = height - k_height + 1
    out_width = width - k_width + 1
    output = np.zeros((out_height, out_width, channels), dtype=np.uint8)
    for z in range(channels):
        normalized_kernel = kernel[:, :, z] / np.sum(kernel[:, :, z])
        for y in range(out_height):
            for x in range(out_width):
                window = image[y:y + k_height, x:x + k_width, z]
                output[y, x, z] = np.sum(window * normalized_kernel)
    return output

import numpy as np

def sobel_filter(image: np.ndarray, mode: str, intensity: int = 1, threshold: int = 127) -> np.ndarray | None:
    if len(image.shape) == 3:
        return image

    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]]) * intensity

    sobel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]]) * intensity

    # Pad the image
    padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant')
    out_height, out_width = image.shape[:]
    output = np.zeros((out_height, out_width), dtype=np.uint8)

    match mode:
        case 'vertical':
            def apply_window(cur_window: np.ndarray) -> np.uint8:
                return abs(np.sum(cur_window * sobel_x))

        case 'horizontal':
            def apply_window(cur_window: np.ndarray) -> np.uint8:
                return abs(np.sum(cur_window * sobel_y))

        case 'both':
            def apply_window(cur_window: np.ndarray) -> np.uint8:
                gx = np.sum(cur_window * sobel_x)
                gy = np.sum(cur_window * sobel_y)
                return np.uint8(np.sqrt(gx ** 2 + gy ** 2))
        case _:
            return None

    for y in range(out_height):
        for x in range(out_width):
            window = padded_image[y:y + 3, x:x + 3]
            val = apply_window(window)
            if val >= threshold:
                output[y, x] = np.uint8(255)
            if val < threshold:
                output[y, x] = np.uint8(0)
    return output.astype(np.uint8)


def laplace_filter(image: np.ndarray, intensity: int = 4, threshold: int = 127) -> np.ndarray:
    if len(image.shape) == 3:
        return image

    kernel = np.array([[0,  -1, 0],
                       [-1, intensity, -1],
                       [0,  -1, 0]])
    height, width = image.shape
    output = np.zeros_like(image, dtype=np.uint8)
    padded_image = np.pad(image, pad_width=1, mode='constant', constant_values=0)
    for y in range(height):
        for x in range(width):
            window = padded_image[y:y+3, x:x+3]
            pixelsum = np.sum(window * kernel)
            pixelsum = np.clip(pixelsum, 0, 255)
            if pixelsum >= threshold:
                output[y, x] = np.uint8(255)
            if pixelsum < threshold:
                output[y, x] = np.uint8(0)
    return output


def linear_gray_scaling(image: np.ndarray, c1: float, c2: float) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    height, width = image.shape
    for y in range(height):
        for x in range(width):
            new_gray_value = c2*image[y, x] + c1*c2
            if new_gray_value > np.iinfo(image.dtype).max:
                new_gray_value = np.iinfo(image.dtype).max
            image[y, x] = new_gray_value
    return image

def isodensity_filter(image: np.ndarray, degree: int) -> np.ndarray | None:
    if len(image.shape) == 3:
        return None
    mean_value = stat_ops.mean(image)
    std = stat_ops.std(image)
    height, width = image.shape
    output = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            match degree:
                case 1:
                    if image[y, x] < mean_value-std:
                        output[y, x] = 0
                    if image[y, x] > mean_value+std:
                        output[y, x] = 255
                    if mean_value-std <= image[y, x] <= mean_value+std:
                        output[y, x] = mean_value
                case 2:
                    output[y, x] = 0 if image[y, x] < mean_value else 255
    return output