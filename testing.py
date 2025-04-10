import numpy as np
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops
import matplotlib.pyplot as plt


bgr_img = basic_ops.load_image('images/resized.ppm')
bgr_img = geo_ops.resize_image(bgr_img, 20, 20)

img = basic_ops.load_image('images/resized.ppm')
img = filters.gray_scale_filter(img)

print(bgr_img.shape)
print(img.shape)

image_histogram = stat_ops.histogram(img)
plt.plot(image_histogram)
plt.show()

image_histogram = stat_ops.histogram(bgr_img)
plt.plot(image_histogram)
plt.show()