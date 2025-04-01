import basic_image_operations as basic_ops
import geometrical_image_operations as geo_ops
import filters


def main():

    img = basic_ops.create_image(800, 600, 1, 128)
    basic_ops.show_image(img, 'created pixelmap')

    img_path = "images/aquarium.jpeg"
    img = basic_ops.load_image(image_path=img_path)

    h, w = img.shape[:2]
    resized_img = geo_ops.resize_image(image=img, target_width=int(w/4), target_height=int(h/4))
    basic_ops.show_image(image=resized_img, title='resized image')
    basic_ops.save_image(image=resized_img, save_path="images/resized.ppm")

    img = resized_img

    bw_image = filters.black_white_filter(img, threshold=128)
    basic_ops.show_image(image=bw_image, title='black white image')
    basic_ops.save_image(image=bw_image, save_path="images/black_white.ppm")

    mirrored_vertically_img = geo_ops.mirror_image(image=img, mode='vertical')
    basic_ops.show_image(image=mirrored_vertically_img, title='mirrored vertically')
    basic_ops.save_image(image=mirrored_vertically_img, save_path="images/mirrored_vertically.ppm")

    mirrored__horizontally_img = geo_ops.mirror_image(image=img, mode='horizontal')
    basic_ops.show_image(image=mirrored__horizontally_img, title='mirrored horizontally')
    basic_ops.save_image(mirrored__horizontally_img, save_path="images/mirrored_horizontally.ppmg")

    rotated_img = geo_ops.rotate_image(image=img, direction=1)
    basic_ops.show_image(image=rotated_img, title="rotated image 90 degree clockwise")
    basic_ops.save_image(image=rotated_img, save_path="images/rotated_image.ppm")

    blur_kernel_dim = 15
    blur_kernel_intensity = 40
    blurred_img = filters.blur_filter(image=img, kernel_dim=blur_kernel_dim, kernel_intensity=blur_kernel_intensity)
    basic_ops.show_image(blurred_img, title='blurred image')
    basic_ops.save_image(blurred_img, save_path="images/blurred.ppm")

    gray_scaled_img = filters.gray_scale_filter(image=img)
    basic_ops.show_image(image=gray_scaled_img, title='gray scaled image')
    basic_ops.save_image(image=gray_scaled_img, save_path="images/gray_scaled.ppm")

    square_img = basic_ops.load_image(image_path='images/square.jpeg')
    h, w = square_img.shape[:2]
    square_img = geo_ops.resize_image(square_img, target_width=int(w/3), target_height=int(h/3))
    square_img = filters.gray_scale_filter(image=square_img)
    sobel_vertical_image = filters.sobel_filter(image=square_img, mode='vertical', intensity=1)
    sobel_horizontal_image = filters.sobel_filter(image=square_img, mode='horizontal', intensity=1)
    basic_ops.show_image(image=sobel_vertical_image, title='vertical sobel filter')
    basic_ops.show_image(image=sobel_horizontal_image, title=f'horizontal sobel filter')
    basic_ops.save_image(image=sobel_vertical_image, save_path=f"images/sobel_vertical.pgm")
    basic_ops.save_image(image=sobel_horizontal_image, save_path=f"images/sobel_horizontal.pgm")


if __name__ == "__main__":
    main()