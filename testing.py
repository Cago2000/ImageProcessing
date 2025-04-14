import numpy as np
from matplotlib import pyplot as plt
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops


img = basic_ops.load_image(image_path='images/obama.pgm')
img = filters.gray_scale_filter(image=img)
basic_ops.show_image(image=img)

img = stat_ops.histogram_equalization(image=img)
basic_ops.show_image(image=img)
basic_ops.save_image(image=img, save_path=f"images/histogram_equalized.pgm")