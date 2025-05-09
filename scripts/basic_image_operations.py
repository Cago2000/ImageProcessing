import cv2
import os
import numpy as np

def create_ppm_image(width: int, height: int, name: str, file_format: str):
    magic_number = None
    pixel_data = ''

    dimensions = f'{width} {height}\n'
    max_color = '255\n'
    width, height = 3, 2
    pixels = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 255, 255),
        (0, 0, 0),
    ]

    if file_format == 'ascii':
        magic_number = 'P3\n'
        for r, g, b in pixels:
            pixel_data += f'{r} {g} {b} '

    if file_format == 'binary':
        magic_number = b'P6\n'
        dimensions = f'{width} {height}\n'.encode()
        max_color = '255\n'.encode()
        pixel_data = bytearray()
        for pixel in pixels:
            pixel_data.extend(pixel)

    ppm_data = magic_number + dimensions + max_color + pixel_data
    file_path = f"images/{name}.ppm"
    mode = 'w' if file_format == 'ascii' else 'wb'
    with open(file_path, mode) as f:
        f.write(ppm_data)


def load_ppm_image(image_path: str) -> np.ndarray:
    with open(image_path, 'rb') as f:
        magic_number = f.readline().strip()
        if magic_number != b'P6':
            raise ValueError('Unsupported PPM format (only P6 is supported)')

        dimensions = f.readline()
        while dimensions.startswith(b'#'):
            dimensions = f.readline()
        width, height = map(int, dimensions.strip().split())

        max_color = f.readline()
        while max_color.startswith(b'#'):
            max_color = f.readline()
        max_color = int(max_color.strip())
        if max_color != 255:
            raise ValueError('Only max color value of 255 is supported')

        pixel_data = f.readline()
        rgb_tuples = [tuple(pixel_data[i:i + 3]) for i in range(0, len(pixel_data), 3)]
        rows = [rgb_tuples[i:i + width] for i in range(0, len(rgb_tuples), width)]
        print(rows)
        image = np.array(rows, dtype=np.uint8)
        image = image[..., ::-1] # RGB to BGR
        return image

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
        if len(img.shape) == 3 and image_path.split('.')[-1] == 'pgm':
            img = img[:,:, 0]
        return np.uint8(img)
    else:
        print("Error: File not found.")
        return None

def load_images(folder_path: str, amount: int = 100) -> list[np.ndarray]:
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.ppm', '.pgm')):
            img_path = os.path.join(folder_path, filename)
            img = load_image(img_path)
            if img is not None:
                images.append(img)
            if len(images) >= amount:
                return images
    return images

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
