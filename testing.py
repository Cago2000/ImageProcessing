import numpy as np
import basic_image_operations as basic_ops
import statistical_operations as stat_ops

test_img = np.array([[0, 0, 1, 1, 2, 3],
                     [0, 0, 0, 1, 2, 3],
                     [0, 0, 1, 2, 3, 3],
                     [0, 1, 1, 2, 3, 3],
                     [1, 2, 2, 3, 3, 3],
                     [2, 2, 3, 3, 3, 3]])

counter = stat_ops.co_occurrence(test_img, (lambda image, x, y: image[y, x] == image[y, x + 1] if x + 1 < image.shape[1] else False))
print(counter)