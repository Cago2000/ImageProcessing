import numpy as np
from matplotlib import pyplot as plt
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops


img = basic_ops.load_image(image_path='images/obama.pgm')
img = filters.gray_scale_filter(image=img)
basic_ops.show_image(image=img)

histogram_equalized_img = stat_ops.histogram_equalization(image=img)
basic_ops.show_image(image=histogram_equalized_img)
basic_ops.save_image(image=histogram_equalized_img, save_path=f"images/histogram_equalized.pgm")

brightened_img = stat_ops.gamma_equalization(image=img, gamma=0.3)
basic_ops.show_image(image=brightened_img)
basic_ops.save_image(image=brightened_img, save_path=f"images/brightened_gamma_equalized.pgm")

darkened_img = stat_ops.gamma_equalization(image=img, gamma=3.0)
basic_ops.show_image(image=darkened_img)
basic_ops.save_image(image=darkened_img, save_path=f"images/darkened_gamma_equalized.pgm")

