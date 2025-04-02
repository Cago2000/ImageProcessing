import numpy as np

def resize_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    x_min, x_max = 0, width - 1
    y_min, y_max = 0, height - 1
    reduced_image = np.zeros((target_height, target_width) + image.shape[2:], dtype=image.dtype)
    for m in range(target_width):
        for n in range(target_height):
            x = x_min + (m / (target_width - 1)) * (x_max - x_min)
            y = y_min + (n / (target_height - 1)) * (y_max - y_min)
            reduced_image[n, m] = image[int(y), int(x)]
    return reduced_image

def rotate_image(image: np.ndarray, direction: int = 1) -> np.ndarray:
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    if direction == 1:
        output = np.zeros((width, height, channels), dtype=image.dtype)
        for i in range(height):
            for j in range(width):
                output[j, height - 1 - i] = image[i, j]
        return output

    if direction == -1:
        output = np.zeros((width, height, channels), dtype=image.dtype)
        for i in range(height):
            for j in range(width):
                output[width - 1 - j, i] = image[i, j]
        return output

    if direction in {2, -2}:
        output = np.zeros((height, width, channels), dtype=image.dtype)
        for i in range(height):
            for j in range(width):
                output[height - 1 - i, width - 1 - j] = image[i, j]
        return output
    return image

def mirror_image(image: np.ndarray, mode: str = 'vertical') -> np.ndarray:
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
    match mode:
        case 'vertical':
            output = np.zeros((height, width, channels), dtype=image.dtype)
            for i in range(height):
                for j in range(width):
                    output[i, j] = image[i, width-j-1]
            return output

        case 'horizontal':
            output = np.zeros((height, width, channels), dtype=image.dtype)
            for i in range(height):
                for j in range(width):
                    output[i, j] = image[height-i-1, j]
            return output
    return image