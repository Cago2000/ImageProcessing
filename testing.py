import basic_image_operations as basic_ops
import geometrical_image_operations as geo_ops

img_path = "images/aquarium.jpeg"
img = basic_ops.load_image(image_path=img_path)
img = geo_ops.resize_image(image=img, target_width=500, target_height=400)
degree = 50
rotated_img = geo_ops.rotate_image(image=img, degree=degree)
basic_ops.show_image(image=rotated_img, title=f"rotated image {degree} degree clockwise")
basic_ops.save_image(image=rotated_img, save_path=f"images/{degree}_degree_rotated_image.ppm")