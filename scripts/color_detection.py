import numpy as np

def get_blobs(mask: np.ndarray) -> list:
    label = 1
    labels = np.zeros_like(mask, dtype=np.int32)
    height, width = mask.shape

    def flood_fill(y, x, label):
        stack = [(y, x)]
        coords = []

        while stack:
            cy, cx = stack.pop()
            is_in_bounds = 0 <= cy < height and 0 <= cx < width
            if is_in_bounds and mask[cy, cx] == 1 and labels[cy, cx] == 0:
                labels[cy, cx] = label
                coords.append((cy, cx))
                stack.extend([(cy+1, cx),
                              (cy-1, cx),
                              (cy, cx+1),
                              (cy, cx-1)])
        return coords

    blobs = []
    for y in range(height):
        for x in range(width):
            if mask[y, x] == 1 and labels[y, x] == 0:
                coords = flood_fill(y, x, label)
                blobs.append(coords)
                label += 1
    return blobs