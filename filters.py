import numpy as np

def gray_scale_filter(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    height, width, _ = image.shape
    gray_scaled_image = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            gray_scaled_image[y][x] = int(0.299 * image[y][x][2] + 0.587 * image[y][x][1] + 0.114 * image[y][x][0])
    return gray_scaled_image

def black_white_filter(image: np.ndarray, threshold: int) -> np.ndarray:
    avg_intensity = np.mean(image, axis=-1)
    output = np.where(avg_intensity >= threshold, 255, 0).astype(np.uint8)
    return output

def blur_filter(image: np.ndarray, kernel_dim: int, kernel_intensity: int) -> np.ndarray:
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

def sobel_filter(image: np.ndarray, mode: str, intensity: int = 1) -> np.ndarray:
    match mode:
        case 'vertical': sobel = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]])
        case 'horizontal': sobel = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]])
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