import numpy as np
from matplotlib import pyplot as plt
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops

bgr_img = basic_ops.load_image('images/resized.ppm')
bgr_img = geo_ops.resize_image(bgr_img, 20, 20)

img = basic_ops.load_image('images/resized.ppm')
img = filters.gray_scale_filter(img)

isodensity_filter_degree_one_img = filters.isodensity_filter(img, 1)
isodensity_filter_degree_two_img = filters.isodensity_filter(img, 2)

iso_one_histogram = stat_ops.histogram(isodensity_filter_degree_one_img)
plt.plot(iso_one_histogram)
plt.show()

iso_two_histogram = stat_ops.histogram(isodensity_filter_degree_two_img)
plt.plot(iso_two_histogram)
plt.show()
