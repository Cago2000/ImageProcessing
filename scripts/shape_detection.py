import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull, QhullError
import math

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

def get_contours(binary_image: np.ndarray, angle_tolerance: int = 10, deviation_threshold: float = 10.0) -> list:
    visited = np.zeros_like(binary_image, dtype=bool)
    contours = []
    height, width = binary_image.shape[:2]

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if is_edge(binary_image, y, x) and not visited[y, x]:
                contour = trace_contour(binary_image, visited, y, x)
                #rect = min_area_rect(contour)
                rect = cv2.minAreaRect(contour)

                if rect is None:
                    continue
                _, _, angle = rect
                angle = abs(angle % 180)
                angle = 90 - angle if angle > 90 else angle

                if abs(angle - 45) > angle_tolerance:
                    continue
                contours.append(contour)
    return contours

def min_area_rect(contour):
    if len(contour) < 3:
        return None
    try:
        hull = ConvexHull(contour)
        hull_points = contour[hull.vertices]

        min_area = float('inf')
        best_rect = None

        for i in range(len(hull_points)):
            p1 = hull_points[i]
            p2 = hull_points[(i + 1) % len(hull_points)]
            edge = p2 - p1
            angle = -math.atan2(edge[1], edge[0]) * 180 / np.pi  # Convert radian to degree
            rotated = rotate_points(hull_points, -angle)

            min_x, min_y = np.min(rotated, axis=0)
            max_x, max_y = np.max(rotated, axis=0)
            width = max_x - min_x
            height = max_y - min_y
            area = width * height

            if area < min_area:
                min_area = area
                center_rotated = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2])
                center = rotate_points(np.array([center_rotated]), angle)[0]
                best_rect = (center, height, width, angle)

        return best_rect

    except QhullError:
        return None

def rotate_points(points, angle):
    radians = np.deg2rad(angle)
    rotation_matrix = np.array([
        [np.cos(radians), -np.sin(radians)],
        [np.sin(radians), np.cos(radians)]
    ])
    return np.dot(points, rotation_matrix.T)

def get_rectangle_corners(center, width, height, angle):
    w, h = width / 2, height / 2
    corners = np.array([
        [-w, -h],
        [w, -h],
        [w, h],
        [-w, h]
    ])
    rotated = rotate_points(corners, angle)
    return rotated + center



'''from numpy.linalg import lstsq
def fit_line(contour):
    points = contour.reshape(-1, 2)  # Reshape contour array
    X = np.c_[points[:, 0], np.ones(points.shape[0])]  # Design matrix
    y = points[:, 1]
    slope, intercept = lstsq(X, y, rcond=None)[0]
    return slope, intercept

def deviation_from_line(contour, slope, intercept):
    points = contour.reshape(-1, 2)
    line_y = slope * points[:, 0] + intercept
    deviations = np.abs(points[:, 1] - line_y)
    return np.mean(deviations), np.std(deviations)

def is_straight(contour, deviation_threshold):
    if len(contour) < 5:
        return False

    slope, intercept = fit_line(contour)
    mean_deviation, std_deviation = deviation_from_line(contour, slope, intercept)
    print(mean_deviation)
    return mean_deviation < deviation_threshold and std_deviation < deviation_threshold'''
