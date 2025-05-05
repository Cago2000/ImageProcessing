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
