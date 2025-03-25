import cv2
import os

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