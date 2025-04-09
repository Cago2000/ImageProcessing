import numpy as np
import basic_image_operations as basic_ops
import statistical_operations as stat_ops

test_img = np.array([[0, 0, 1, 1, 2, 3, 4],
                     [0, 0, 0, 1, 2, 3, 5],
                     [0, 0, 1, 2, 3, 3, 4],
                     [0, 1, 1, 3, 3, 3, 4],
                     [1, 2, 2, 3, 3, 3, 3],
                     [2, 2, 3, 3, 3, 3, 2],
                     [2, 2, 4, 4, 5, 6, 3]])

counter = stat_ops.co_occurrence(test_img, (lambda image, x, y: image[y, x] == image[y, x + 1] if x + 1 < image.shape[1] else False))
print(counter)

median_value = stat_ops.median(test_img)
print(median_value)

img = basic_ops.load_image('images/resized.ppm')
mean_value = stat_ops.mean(img)
print(mean_value)

mean_value = stat_ops.mean(test_img)
print(mean_value)