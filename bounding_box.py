import numpy as np
import colors

class BoundingBox:
    def __init__(self, y: int, x: int, height: int, width: int, area: int, label: str):
        self.center_y = y
        self.center_x = x
        self.box_height = height
        self.box_width = width
        self.box_area = area
        self.label = label

    def __repr__(self):
        return (f"BoundingBox(center_y={self.center_y}, center_x={self.center_x}, "
                f"height={self.box_height}, width={self.box_width}, area={self.box_area}, label='{self.label}')")

def draw_bounding_boxes(image: np.ndarray, blobs: list, min_box_area: int, box_color: list[int]) -> tuple:
    result = image.copy()
    bounding_boxes = []
    for blob in blobs:
        ys, xs = zip(*blob)
        top, left = min(ys), min(xs)
        bottom, right = max(ys), max(xs)

        width = right - left + 1
        height = bottom - top + 1
        area = width * height

        if area < min_box_area:
            continue

        result[top, left:right+1] = box_color
        result[bottom, left:right+1] = box_color

        result[top:bottom+1, left] = box_color
        result[top:bottom+1, right] = box_color

        bounding_box_obj = BoundingBox(((top + bottom) // 2), ((right + left) // 2), height, width, area, colors.get_str_from_color(box_color))
        bounding_boxes.append(bounding_box_obj)
    return result, bounding_boxes