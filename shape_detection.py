import numpy as np
import cv2
import bounding_box

def is_edge(binary_img: np.ndarray, y: int, x: int) -> bool:
    height, width = binary_img.shape[:2]
    if binary_img[y, x] != 255:
        return False
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        is_in_bounds = 0 <= ny < height and 0 <= nx < width
        if is_in_bounds and binary_img[ny, nx] == 0:
            return True
    return False

def trace_contour(binary_image: np.ndarray, visited: np.ndarray, y: int, x: int) -> np.ndarray:
    height, width = visited.shape[:2]
    directions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                  (1, 0), (1, -1), (0, -1), (-1, -1)]
    contour = []
    stack = [(y, x)]
    visited[y, x] = True

    while stack:
        cy, cx = stack.pop()
        contour.append((cy, cx))
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            is_in_bounds = 0 <= ny < height and 0 <= nx < width
            if is_in_bounds and not visited[ny, nx] and is_edge(binary_image, ny, nx):
                visited[ny, nx] = True
                stack.append((ny, nx))
                break
    return np.array(contour)

def get_contours(binary_image: np.ndarray, angle_tolerance: int = 10) -> list:
    visited = np.zeros_like(binary_image, dtype=bool)
    contours = []
    height, width = binary_image.shape[:2]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if is_edge(binary_image, y, x) and not visited[y, x]:
                contour = trace_contour(binary_image, visited, y, x)
                rect = cv2.minAreaRect(contour)
                _, _, angle = rect

                angle = abs(angle % 180)
                angle = 90 - angle if angle > 90 else angle  # normalize to 0–90°

                if abs(angle - 45) > angle_tolerance:
                    continue
                contours.append(contour)
    return contours
