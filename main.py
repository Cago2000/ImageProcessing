import image_operations as img_ops
from image_operations import show_image, black_white_image


def main():
    img_path = "images/aquarium.jpeg"
    img = img_ops.load_image(image_path=img_path)

    h, w = img.shape[:2]
    resized_img = img_ops.resize_image(image=img, target_width=int(w/4), target_height=int(h/4))
    img_ops.show_image(image=resized_img, title='resized image')
    img_ops.save_image(image=resized_img, save_path="images/resized.jpeg")

    img = resized_img

    bw_image = black_white_image(img, threshold=128)
    img_ops.show_image(image=bw_image, title='black white image')
    img_ops.save_image(image=bw_image, save_path="images/black_white.jpeg")

    mirrored_vertically_img = img_ops.mirror_image(image=img, mode='vertical')
    img_ops.show_image(image=mirrored_vertically_img, title='mirrored vertically')
    img_ops.save_image(image=mirrored_vertically_img, save_path="images/mirrored_vertically.jpeg")

    mirrored__horizontally_img = img_ops.mirror_image(image=img, mode='horizontal')
    img_ops.show_image(image=mirrored__horizontally_img, title='mirrored horizontally')
    img_ops.save_image(mirrored__horizontally_img, save_path="images/mirrored_horizontally.jpeg")

    rotated_img = img_ops.rotate_image(image=img, direction=1)
    img_ops.show_image(image=rotated_img, title="rotated image 90 degree clockwise")
    img_ops.save_image(image=rotated_img, save_path="images/rotated_image.jpeg")

    blur_kernel_dim = 15
    blur_kernel_intensity = 40
    blurred_img = img_ops.blur_filter(image=img, kernel_dim=blur_kernel_dim, kernel_intensity=blur_kernel_intensity)
    img_ops.show_image(blurred_img, title='blurred image')
    img_ops.save_image(blurred_img, save_path="images/blurred.jpeg")

    gray_scaled_img = img_ops.gray_scale_image(image=img)
    img_ops.show_image(image=gray_scaled_img, title='gray scaled image')
    img_ops.save_image(image=gray_scaled_img, save_path="images/gray_scaled.jpeg")

    square_img = img_ops.load_image(image_path='images/square.jpeg')
    h, w = square_img.shape[:2]
    square_img = img_ops.resize_image(square_img, target_width=int(w/3), target_height=int(h/3))
    square_img = img_ops.gray_scale_image(image=square_img)
    sobel_vertical_image = img_ops.sobel_filter(image=square_img, mode='vertical', intensity=1)
    sobel_horizontal_image = img_ops.sobel_filter(image=square_img, mode='horizontal', intensity=1)
    img_ops.show_image(image=sobel_vertical_image, title='vertical sobel filter')
    img_ops.show_image(image=sobel_horizontal_image, title=f'horizontal sobel filter')
    img_ops.save_image(image=sobel_vertical_image, save_path=f"images/sobel_vertical.jpeg")
    img_ops.save_image(image=sobel_horizontal_image, save_path=f"images/sobel_horizontal.jpeg")


if __name__ == "__main__":
    main()