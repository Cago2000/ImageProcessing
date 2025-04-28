import numpy as np
from matplotlib import pyplot as plt
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops
import traffic_sign_detection as tsd
import cv2

img = basic_ops.load_image(image_path='traffic_sign_images/false_01.jpg')
img = geo_ops.resize_image(image=img, target_height=img.shape[0]//8, target_width=img.shape[1]//8)
img = filters.gray_scale_filter(image=img)
img = filters.sobel_filter(image=img, mode='both')
img = filters.black_white_filter(image=img, threshold=127)
basic_ops.show_image(image=img)
