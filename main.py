import cv2
import image_operations as img_ops
from image_operations import gray_scale_image, show_image, save_image, resize_image


def main():
    img_path = "images/aquarium.jpeg"  # Change this to your image path
    img = img_ops.load_image(img_path)

    img = gray_scale_image(img)
    show_image(img)
    save_image(img,"images/gray_scaled.jpeg")

    h, w = img.shape[:2]
    img = resize_image(img, int(w/4), int(h/4))
    show_image(img)
    save_image(img,"images/resized.jpeg")
    
if __name__ == "__main__":
    main()