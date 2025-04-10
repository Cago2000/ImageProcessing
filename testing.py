import basic_image_operations as basic_ops
import filters
import geometrical_image_operations as geo_ops
import statistical_operations as stat_ops


bgr_img = basic_ops.load_image('images/resized.ppm')
bgr_img = geo_ops.resize_image(bgr_img, 20, 20)

img = basic_ops.load_image('images/resized.ppm')
img = filters.gray_scale_filter(img)

entropy_value = stat_ops.entropy(img)
print(entropy_value)