import cv2
import os
import numpy as np

def load_image(image_path):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to load image.")
            return None
        print(f"Image loaded from {image_path}")
        return img
    else:
        print("Error: File not found.")
        return None

def save_image(image, save_path):
    try:
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def delete_image(image_path):
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Image deleted at {image_path}")
    else:
        print("Error: File not found.")

def show_image(image):
    if image is not None:
        print(f"Image displayed")
        cv2.imshow("Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: File not found.")

def gray_scale_image(image):
    if image is not None:
        image = np.array(
            [list(map(lambda bgr: int(0.299 * bgr[2] + 0.587 * bgr[1] + 0.114 * bgr[0]), row)) for row in image])
        return np.uint8(image)
    else:
        return image

def resize_image(image, target_width, target_height):
    height, width = image.shape[:2]
    x_min, x_max = 0, width - 1
    y_min, y_max = 0, height - 1
    reduced_image = np.zeros((target_height, target_width) + image.shape[2:], dtype=image.dtype)
    for m in range(target_width):
        for n in range(target_height):
            x = x_min + (m / (target_width - 1)) * (x_max - x_min)
            y = y_min + (n / (target_height - 1)) * (y_max - y_min)
            reduced_image[n, m] = image[int(y), int(x)]
    return reduced_image