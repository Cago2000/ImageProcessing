from typing import Callable
import numpy as np

def bgr_to_hsv(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[2], rgb[1], rgb[0]
    r, g, b = r / 255, g / 255, b / 255
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    delta = max_c - min_c

    if delta == 0:
        h = 0
    else:
        match max_c:
            case x if x == r:
                h = 60 * (((g - b) / delta) % 6)
            case x if x == g:
                h = 60 * (((b - r) / delta) + 2)
            case x if x == b:
                h = 60 * (((r - g) / delta) + 4)
    s = 0 if max_c == 0 else delta / max_c
    v = max_c
    return np.array([h, s, v])

def get_color_from_function(color_function: Callable) -> list[int]:
    match color_function:
        case func if func is is_strong_red:
            return [0, 0, 255]
        case func if func is is_strong_green:
            return [0, 255, 0]
        case func if func is is_strong_yellow:
            return [0, 255, 255]  # Yellow in BGR
        case func if func is is_strong_blue:
            return [255, 0, 0]
        case _:
            return [0, 0, 0]


def get_str_from_color(color: list[int]) -> str:
    match color:
        case [0, 0, 255]:
            return 'r'
        case [0, 255, 0]:
            return 'g'
        case[255, 0, 0]:
            return 'b'
        case [255, 255, 0]:
            return 'y'
        case _:
            return 'default'


def get_mask(image: np.ndarray, color_function: Callable):
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.bool)
    for y in range(height):
        for x in range(width):
            hsv = bgr_to_hsv(image[y, x])
            h, s, v = hsv[0], hsv[1], hsv[2]
            if color_function(h, s, v):
                mask[y, x] = 1
    return mask

def is_strong_red(h: int, s: np.float64, v: np.float64) -> bool:
    is_hue_red = h >= 345 or h <= 15
    is_saturated = s >= 0.3
    is_bright_enough = v >= 0.1
    return is_hue_red and is_saturated and is_bright_enough

def is_strong_green(h: int, s: np.float64, v: np.float64) -> bool:
    is_hue_green = 105 <= h <= 135
    is_saturated = s >= 0.3
    is_bright_enough = v >= 0.1
    return is_hue_green and is_saturated and is_bright_enough

def is_strong_blue(h: int, s: np.float64, v: np.float64) -> bool:
    is_hue_blue = 225 <= h <= 255
    is_saturated = s >= 0.3
    is_bright_enough = v >= 0.1
    return is_hue_blue and is_saturated and is_bright_enough


def is_strong_yellow(h: int, s: np.float64, v: np.float64) -> bool:
    is_hue_yellow = 40 <= h <= 70
    is_saturated = s >= 0.3
    is_bright_enough = v >= 0.1
    return is_hue_yellow and is_saturated and is_bright_enough


