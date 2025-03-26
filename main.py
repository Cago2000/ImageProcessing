import image_operations as img_ops
from image_operations import gray_scale_image, show_image, save_image, resize_image, load_image


def main():
    img_path = "images/aquarium.jpeg"  # Change this to your image path
    img = img_ops.load_image(image_path=img_path)

    h, w = img.shape[:2]
    resized_img = resize_image(image=img, target_width=int(w/4), target_height=int(h/4))
    show_image(image=resized_img, title='resized image')
    save_image(image=resized_img, save_path="images/resized.jpeg")

    blur_kernel_dim = 15
    blur_kernel_intensity = 40
    blurred_img = img_ops.blur_filter(image=img, kernel_dim=blur_kernel_dim, kernel_intensity=blur_kernel_intensity)
    show_image(blurred_img, title='blurred image')
    save_image(blurred_img, save_path="images/blurred.jpeg")

    gray_scaled_img = gray_scale_image(image=img)
    show_image(image=gray_scaled_img, title='gray scaled image')
    save_image(image=gray_scaled_img, save_path="images/gray_scaled.jpeg")

    square_img = load_image(image_path='images/square.png')
    square_img = gray_scale_image(image=square_img)
    sobel_vertical_image = img_ops.sobel_filter(image=square_img, mode='vertical', intensity=1)
    sobel_horizontal_image = img_ops.sobel_filter(image=square_img, mode='horizontal', intensity=1)
    show_image(image=sobel_vertical_image, title='vertical sobel filter')
    show_image(image=sobel_horizontal_image, title=f'horizontal sobel filter')
    save_image(image=sobel_vertical_image, save_path=f"images/sobel_vertical.jpg")
    save_image(image=sobel_horizontal_image, save_path=f"images/sobel_horizontal.jpg")


if __name__ == "__main__":
    main()