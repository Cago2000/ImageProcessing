import cv2
import os
import numpy as np

def create_image(width: int, height: int, channels: int, gray_value: int) -> np.ndarray:
    image = np.zeros([height, width, channels], dtype=np.uint8)
    image[:] = gray_value
    return image

def create_image_with_gradient(width: int, height: int, brightness: int) -> np.ndarray:
    image = np.zeros([height, width, 3], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            blue = int(brightness * (x / width))
            green = int(brightness * (y / height))
            red = int(brightness * ((x + y) / (width + height)))
            image[y, x] = (blue, green, red)
    return image

def load_image(image_path: str) -> np.ndarray | None:
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print("Error: Unable to load image.")
            return None
        print(f"Image loaded from {image_path}")
        return np.uint8(img)
    else:
        print("Error: File not found.")
        return None

def save_image(image: np.ndarray, save_path: str) -> None:
    try:
        cv2.imwrite(save_path, image)
        print(f"Image saved at {save_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def delete_image(image_path: str) -> None:
    if os.path.exists(image_path):
        os.remove(image_path)
        print(f"Image deleted at {image_path}")
    else:
        print("Error: File not found.")

def show_image(image: np.ndarray, title: str = 'Image') -> None:
    if image is not None:
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(f"Image displayed")
    else:
        print("Error: File not found.")
