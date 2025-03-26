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
        return np.uint8(img)
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

def show_image(image, title='Image'):
    if image is not None:
        print(f"Image displayed")
        cv2.imshow(title, image)
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


def rotate_image(image, direction=1):
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1

    if direction == 1:
        output = np.zeros((width, height, channels), dtype=image.dtype)
        for i in range(height):
            for j in range(width):
                output[j, height - 1 - i] = image[i, j]
        return output

    if direction == -1:
        output = np.zeros((width, height, channels), dtype=image.dtype)
        for i in range(height):
            for j in range(width):
                output[width - 1 - j, i] = image[i, j]
        return output

    if direction in {2, -2}:
        output = np.zeros((height, width, channels), dtype=image.dtype)
        for i in range(height):
            for j in range(width):
                output[height - 1 - i, width - 1 - j] = image[i, j]
        return output
    return image

def mirror_image(image, mode='vertical'):
    if len(image.shape) == 3:
        height, width, channels = image.shape
    else:
        height, width = image.shape
        channels = 1
    match mode:
        case 'vertical':
            output = np.zeros((height, width, channels), dtype=image.dtype)
            for i in range(height):
                for j in range(width):
                    print(j)
                    output[i, j] = image[i, width-j-1]
            return output

        case 'horizontal':
            output = np.zeros((height, width, channels), dtype=image.dtype)
            for i in range(height):
                for j in range(width):
                    output[i, j] = image[height-i-1, j]
            return output
    return image

def blur_filter(image, kernel_dim, kernel_intensity):
    kernel = np.ones((kernel_dim, kernel_dim, 3)) / kernel_intensity
    pad_size = kernel.shape[0] // 2
    image = np.pad(
        image,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='constant'
    )
    height, width, channels = image.shape
    k_height, k_width, k_channels = kernel.shape
    out_height = height - k_height + 1
    out_width = width - k_width + 1
    output = np.zeros((out_height, out_width, channels), dtype=np.uint8)
    for z in range(channels):
        normalized_kernel = kernel[:, :, z] / np.sum(kernel[:, :, z])
        for y in range(out_height):
            for x in range(out_width):
                window = image[y:y + k_height, x:x + k_width, z]
                output[y, x, z] = np.sum(window * normalized_kernel)
    return output


def sobel_filter(image, mode, intensity=1):
    match mode:
        case 'vertical': sobel = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
        case 'horizontal': sobel = np.array([[-1, -2, -1],[0, 0,0],[1, 2, 1]])
        case _: return image
    sobel *= intensity
    padded_image = np.pad(image,((1, 1), (1, 1)),mode='constant')
    height, width = padded_image.shape
    k_height, k_width = sobel.shape
    out_height = height - k_height + 1
    out_width = width - k_width + 1
    output = np.zeros((out_height, out_width), dtype=np.uint8)
    for y in range(out_height):
        for x in range(out_width):
            window = padded_image[y:y + k_height, x:x + k_width]
            output[y, x] = np.abs(np.sum(window * sobel))
    return output