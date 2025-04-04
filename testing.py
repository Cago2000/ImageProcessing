import numpy as np
import basic_image_operations as basic_ops
import statistical_operations as stat_ops

img = basic_ops.create_image_with_gradient(20, 20, 255)
img[10, 10] = 255
img[10, 11] = 255
img[10, 12] = 255
img[10, 13] = 200
img[10, 14] = 200

relation = lambda image, y, x: np.array_equal(image[y, x], image[y, x+1]) if x+1 < image.shape[1] else False
result = stat_ops.co_occurrence(image=img, relation_function=relation)
print(result)
basic_ops.show_image(img)