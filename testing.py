import basic_image_operations as basic_ops
import geometrical_image_operations as geo_ops

img_path = "images/aquarium.jpeg"
img = basic_ops.load_image(image_path=img_path)

img = geo_ops.resize_image(image=img, target_width=600, target_height=400)
basic_ops.show_image(image=img, title='resized image')

degree = 45
rotated_img = geo_ops.rotate_image(image=img, degree=degree)
basic_ops.show_image(image=rotated_img, title=f"rotated image {degree} degree clockwise")

degree = 59
rotated_img = geo_ops.rotate_image(image=img, degree=degree)
basic_ops.show_image(image=rotated_img, title=f"rotated image {degree} degree clockwise")

degree = 147
rotated_img = geo_ops.rotate_image(image=img, degree=degree)
basic_ops.show_image(image=rotated_img, title=f"rotated image {degree} degree clockwise")

degree = 360
rotated_img = geo_ops.rotate_image(image=img, degree=degree)
basic_ops.show_image(image=rotated_img, title=f"rotated image {degree} degree clockwise")
