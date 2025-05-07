import numpy as np

class BoundingBox:
    def __init__(self, y: int, x: int, corners: list[tuple[int, int]],height: int, width: int, area: int, box_color: list[int], image_index: int):
        self.center_y = y
        self.center_x = x
        self.box_corners = corners
        self.box_height = height
        self.box_width = width
        self.box_area = area
        self.box_color = box_color
        self.image_index = image_index

    def __repr__(self):
        return (f"BoundingBox(center_y={self.center_y}, center_x={self.center_x}, box_corners={self.box_corners}, "
                f"height={self.box_height}, width={self.box_width}, area={self.box_area}, box_color={self.box_color}, "
                f"image_index={self.image_index})")

def create_bounding_boxes(blobs: list, min_box_area: int, box_color: list[int], image_index: int) -> list[BoundingBox]:
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

        center_y = (top + bottom) // 2
        center_x = (left + right) // 2
        box_corners = [top, left, bottom, right]

        bounding_box_obj = BoundingBox(center_y, center_x, box_corners, height, width, area, box_color, image_index)
        bounding_boxes.append(bounding_box_obj)
    return bounding_boxes

def draw_bounding_box(bounding_box: BoundingBox, image: np.ndarray) -> np.ndarray:
    top, left, bottom, right = (bounding_box.box_corners[0],
                                bounding_box.box_corners[1],
                                bounding_box.box_corners[2],
                                bounding_box.box_corners[3])
    box_color = bounding_box.box_color
    image[top, left:right+1] = box_color
    image[bottom, left:right+1] = box_color
    image[top:bottom+1, left] = box_color
    image[top:bottom+1, right] = box_color
    return image