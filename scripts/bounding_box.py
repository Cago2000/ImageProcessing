import numpy as np

class BoundingBox:
    def __init__(self, y: int, x: int, corners: list[int], height: int, width: int, area: int, box_color: list[int], image_index: int):
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

def create_bounding_boxes(blobs: list, image_index: int, min_box_area: int, max_box_area: int, box_color: list[int]) -> list[BoundingBox] | None:
    bounding_boxes = []
    for blob in blobs:
        bounding_box_obj = create_bounding_box(blob, image_index, min_box_area, max_box_area, box_color)
        if bounding_box_obj is not None:
            bounding_boxes.append(bounding_box_obj)
    return bounding_boxes

def create_bounding_box(blob: np.ndarray, image_index: int, min_box_area: int, max_box_area: int, box_color: list[int]) -> BoundingBox | None:
    y_vals, x_vals = zip(*blob)
    top, left = min(y_vals), min(x_vals)
    bottom, right = max(y_vals), max(x_vals)
    width = right - left + 1
    height = bottom - top + 1
    area = width * height

    if min_box_area > area or area > max_box_area:
        return None

    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > 1.75:
        return None

    center_y = (top + bottom) // 2
    center_x = (left + right) // 2
    box_corners = [top, left, bottom, right]

    return BoundingBox(center_y, center_x, box_corners, height, width, area, box_color, image_index)


def draw_bounding_box(bounding_box: BoundingBox, image: np.ndarray) -> np.ndarray:
    top, left, bottom, right = (bounding_box.box_corners[0],
                                bounding_box.box_corners[1],
                                bounding_box.box_corners[2],
                                bounding_box.box_corners[3])

    height, width = image.shape[:2]
    top = max(0, min(height - 1, top))
    bottom = max(0, min(height - 1, bottom))
    left = max(0, min(width - 1, left))
    right = max(0, min(width - 1, right))

    box_color = bounding_box.box_color
    image[top, left:right+1] = box_color
    image[bottom, left:right+1] = box_color
    image[top:bottom+1, left] = box_color
    image[top:bottom+1, right] = box_color
    return image

def fuse_bounding_box_matches(boxes1: list[BoundingBox], boxes2: list[BoundingBox], max_deviation: int) -> list[BoundingBox]:
    new_boxes = []
    for box1 in boxes1:
        for box2 in boxes2:
            if abs(box1.center_y-box2.center_y) >= max_deviation or abs(box1.center_x-box2.center_x) >= max_deviation: #similar enough
                continue
            new_box_corners = []
            for box1_corner, box2_corner in zip(box1.box_corners, box2.box_corners):
                new_box_corners.append((box1_corner+box2_corner)//2)
            new_box_center_y, new_box_center_x = (box1.center_y+box2.center_y)//2, (box1.center_x+box2.center_x)//2
            new_box_height, new_box_width = (box1.box_height+box2.box_height)//2, (box1.box_width+box2.box_width)//2
            new_box_area = new_box_height*new_box_width

            new_box_color = [255, 255, 255]
            if box1.box_color != [255, 255, 255]:
                new_box_color = box1.box_color
            if box2.box_color != [255, 255, 255]:
                new_box_color = box2.box_color

            new_box_image_index = box1.image_index
            new_bounding_box_obj = BoundingBox(new_box_center_y, new_box_center_x, new_box_corners, new_box_height, new_box_width, new_box_area, new_box_color, new_box_image_index)
            if new_bounding_box_obj is not None:
                new_boxes.append(new_bounding_box_obj)
    return new_boxes

def merge_duplicate_boxes(boxes: list[BoundingBox], max_deviation: int) -> list[BoundingBox]:
    merged_boxes = []
    visited = set()
    for box1 in boxes:
        if box1 in visited:
            continue
        similar_boxes = [box1]
        visited.add(box1)

        for box2 in boxes:
            if box2 in visited or box1 is box2:
                continue
            if (abs(box1.center_y - box2.center_y) <= max_deviation and
                abs(box1.center_x - box2.center_x) <= max_deviation):
                similar_boxes.append(box2)
                visited.add(box2)

        if similar_boxes:
            avg_corners = []
            for corners in zip(*(b.box_corners for b in similar_boxes)):
                avg_corner = sum(corners) // len(similar_boxes)
                avg_corners.append(avg_corner)

            center_y = sum(b.center_y for b in similar_boxes) // len(similar_boxes)
            center_x = sum(b.center_x for b in similar_boxes) // len(similar_boxes)
            height = sum(b.box_height for b in similar_boxes) // len(similar_boxes)
            width = sum(b.box_width for b in similar_boxes) // len(similar_boxes)
            area = height * width
            image_index = similar_boxes[0].image_index

            box_color = [255, 255, 255]
            for box in similar_boxes:
                if box.box_color != [255, 255 ,255]:
                    box_color = box.box_color
                    break

            merged_box = BoundingBox(center_y, center_x, avg_corners, height, width, area, box_color, image_index)
            merged_boxes.append(merged_box)

    return merged_boxes