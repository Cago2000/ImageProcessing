import numpy as np
from matplotlib import pyplot as plt
import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops
import traffic_sign_detection as tsd
import cv2

imgs = basic_ops.load_images(folder_path='traffic_sign_images', amount=10)
templates = basic_ops.load_images(folder_path='traffic_sign_templates', amount=5)

for i, img in enumerate(imgs):
    imgs[i] = geo_ops.resize_image(image=img,target_width=img.shape[1]//4, target_height=img.shape[0]//4)
    for template in templates:
        for scale in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 75, 90, 100]:
            resized_template = geo_ops.resize_image(image=template, target_width=img.shape[1]//scale, target_height=img.shape[0]//scale)
            boxes = tsd.match_template(image=imgs[i], template=resized_template, threshold=0.5)
            for top_left, bottom_right in boxes:
                cv2.rectangle(img=imgs[i], pt1=top_left, pt2=bottom_right, color=(0, 255, 0), thickness=1)

for i, img in enumerate(imgs):
    basic_ops.show_image(image=img, title=f'Image #{i+1}')



