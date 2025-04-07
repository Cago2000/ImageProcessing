import math
import numpy as np

def resize_image(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    x_min, x_max = 0, width - 1
    y_min, y_max = 0, height - 1
    resized_image = np.zeros((target_height, target_width) + image.shape[2:], dtype=image.dtype)
    for m in range(target_width):
        for n in range(target_height):
            x = x_min + (m / (target_width - 1)) * (x_max - x_min)
            y = y_min + (n / (target_height - 1)) * (y_max - y_min)
            resized_image[n, m] = image[int(y), int(x)]
    return resized_image


def rotate_image(image: np.ndarray, degree: int = 90) -> np.ndarray:
    if degree == 360:
        return image

    rad = math.radians(degree)

    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    cos_theta = abs(math.cos(rad))
    sin_theta = abs(math.sin(rad))
    new_width = int(width * cos_theta + height * sin_theta)
    new_height = int(width * sin_theta + height * cos_theta)

    output = np.zeros((new_height, new_width, channels), dtype=image.dtype)

    original_cx, original_cy = width / 2, height / 2
    new_cx, new_cy = new_width / 2, new_height / 2

    for y in range(height):
        for x in range(width):
            x_shifted, y_shifted = x - original_cx, y - original_cy
            new_x = int(math.cos(rad) * x_shifted - math.sin(rad) * y_shifted + new_cx)
            new_y = int(math.sin(rad) * x_shifted + math.cos(rad) * y_shifted + new_cy)

            if 0 <= new_x < new_width and 0 <= new_y < new_height:
                output[new_y, new_x] = image[y, x]

    for y in range(new_height):
        for x in range(new_width):
            pixel_is_not_black = np.sum(output[y, x].astype(np.int16)) != 0
            if pixel_is_not_black:
                continue

            x_shifted, y_shifted = x - new_cx, y - new_cy
            original_x = int(math.cos(-rad) * x_shifted - math.sin(-rad) * y_shifted + original_cx)
            original_y = int(math.sin(-rad) * x_shifted + math.cos(-rad) * y_shifted + original_cy)

            pixel_not_in_original_image = original_x < 0 or original_y < 0 or original_x >= width or original_y >= height
            if pixel_not_in_original_image:
                continue

            pixels = []
            for a, b in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                out_of_bounds = not (0 <= x + a < new_width and 0 <= y + b < new_height)
                if out_of_bounds:
                    continue
                pixel_is_black = np.sum(output[y + b, x + a].astype(np.int16)) == 0
                if pixel_is_black:
                    continue
                pixels.append(output[y + b, x + a].astype(np.int16))

            if pixels:
                output[y, x] = np.uint8(sum(pixels) / len(pixels))

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