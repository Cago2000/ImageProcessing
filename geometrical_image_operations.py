import math
import basic_image_operations as basic_ops
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


def rotate_image(image: np.ndarray, degree: int = 90) -> np.ndarray:
    if degree == 360:
        return image

    degree = math.radians(degree)

    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    output = np.zeros((height, width, channels), dtype=image.dtype)
    cx, cy = width // 2, height // 2

    for y in range(height):
        for x in range(width):
            x_shifted, y_shifted = x - cx, y - cy
            new_x = int(math.cos(degree) * x_shifted - math.sin(degree) * y_shifted + cx)
            new_y = int(math.sin(degree) * x_shifted + math.cos(degree) * y_shifted + cy)
            if 0 <= new_x < width and 0 <= new_y < height:
                output[new_y, new_x] = image[y, x]

    for y in range(height):
        for x in range(width):
            if np.sum(output[y, x].astype(np.int16)) != 0:
                continue

            x_shifted, y_shifted = x - cx, y - cy
            original_x = int(math.cos(-degree) * x_shifted - math.sin(-degree) * y_shifted + cx)
            original_y = int(math.sin(-degree) * x_shifted + math.cos(-degree) * y_shifted + cy)

            if not (0 <= original_x < width and 0 <= original_y < height):
                continue

            pixels = []

            for a in [-1, 0, 1]:
                for b in [-1, 0, 1]:
                    if a == 0 and b == 0:
                        continue
                    if x + a < 0 or y + b < 0 or x + a >= width or y + b >= height:
                        continue
                    if np.sum(output[y + b, x + a].astype(np.int16)) == 0:
                        continue
                    if 0 <= x + a < width and 0 <= y + b < height:
                        pixels.append(output[y + b, x + a].astype(np.int16))

            if len(pixels) > 0:
                pixel_sum = np.uint8(sum(pixels)/len(pixels))
                output[y, x] = pixel_sum
    return output


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