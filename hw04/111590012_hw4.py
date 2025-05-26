import cv2
import os
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_result_dir():
    result_dir = os.path.join(CURRENT_DIR, 'result_img')
    os.makedirs(result_dir, exist_ok=True)


def get_all_test_img_path():
    test_dir = os.path.join(CURRENT_DIR, 'test_img')
    jpg_files = [os.path.join(test_dir, f)
                 for f in os.listdir(test_dir) if f.endswith('.jpg')]
    return jpg_files


def convert_to_gray(image):
    B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray_image = (0.3*R + 0.59*G + 0.11*B).astype('uint8')
    return gray_image


def gaussian(image, kernel_size=5, sigma=1.0):
    kernel = [[0]*kernel_size for _ in range(kernel_size)]
    mean = kernel_size // 2

    for x in range(kernel_size):
        for y in range(kernel_size):
            dx = x - mean
            dy = y - mean
            kernel[x][y] = (1/(2*np.pi*sigma**2)) * \
                np.exp(-(dx**2 + dy**2) / (2*sigma**2))

    H, W = image.shape[:2]
    offset = kernel_size // 2
    gaussian_image = np.zeros_like(image)
    for i in range(H):
        for j in range(W):
            sum_val = 0
            weight_sum = 0.0
            for kx in range(kernel_size):
                for ky in range(kernel_size):
                    x = i + kx - offset
                    y = j + ky - offset
                    if 0 <= x < H and 0 <= y < W:
                        weight = kernel[kx][ky]
                        sum_val += image[x, y] * weight
                        weight_sum += weight
            gaussian_image[i, j] = sum_val / weight_sum

    return gaussian_image.astype(np.uint8)


def sobel(image):
    H, W = image.shape[:2]
    G = np.zeros_like(image, dtype=np.float32)
    theta = np.zeros_like(image, dtype=np.float32)
    sobel_x = [[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]]
    sobel_y = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]

    for i in range(1, H-1):
        for j in range(1, W-1):
            gx = 0
            gy = 0
            for kx in range(3):
                for ky in range(3):
                    x = i + kx - 1
                    y = j + ky - 1
                    if 0 <= x < H and 0 <= y < W:
                        gx += image[x, y] * sobel_x[kx][ky]
                        gy += image[x, y] * sobel_y[kx][ky]
            G[i, j] = np.sqrt(gx**2 + gy**2)
            theta[i, j] = np.arctan2(gy, gx)

    G = (G / np.max(G) * 255).astype(np.uint8)
    theta = (np.degrees(theta) + 180) % 180

    return G, theta


def NMS(G, theta):
    H, W = G.shape
    result = np.zeros_like(G)

    for i in range(1, H-1):
        for j in range(1, W-1):
            angle = theta[i, j]

            first = 255
            second = 255
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                first = G[i, j+1]
                second = G[i, j-1]
            elif (22.5 <= angle < 67.5):
                first = G[i+1, j-1]
                second = G[i-1, j+1]
            elif (67.5 <= angle < 112.5):
                first = G[i+1, j]
                second = G[i-1, j]
            elif (112.5 <= angle < 157.5):
                first = G[i-1, j-1]
                second = G[i+1, j+1]
            if G[i, j] >= first and G[i, j] >= second:
                result[i, j] = G[i, j]
            else:
                result[i, j] = 0
    return result


def double_threshold(image, low_ratio=0.05, high_ratio=0.15):
    max_value = np.max(image)
    high = max_value * high_ratio
    low = max_value * low_ratio
    result = np.zeros_like(image)
    H, W = image.shape
    for i in range(H):
        for j in range(W):
            if image[i, j] >= high:
                result[i, j] = 255
            elif image[i, j] >= low:
                result[i, j] = 128
            else:
                result[i, j] = 0
    return result


def hysteresis(image, week=128, strong=255):
    H, W = image.shape
    result = np.zeros_like(image)
    for i in range(H):
        for j in range(W):
            if image[i, j] == strong:
                result[i, j] = strong
            elif image[i, j] == week:
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if 0 <= i + dx < H and 0 <= j + dy < W:
                            if image[i+dx, j+dy] == strong:
                                result[i, j] = strong
                                break

    return result


def main():
    create_result_dir()
    image_paths = get_all_test_img_path()

    for image_path in image_paths:
        origin_image = cv2.imread(image_path)
        gray_image = convert_to_gray(origin_image)
        gaussian_image = gaussian(gray_image)
        sobel_image, theta = sobel(gaussian_image)
        nms_image = NMS(sobel_image, theta)
        thresholded_image = double_threshold(nms_image)
        canny = hysteresis(thresholded_image)
        # save result
        prefix = os.path.join(CURRENT_DIR, 'result_img',
                              os.path.splitext(os.path.basename(image_path))[0])
        cv2.imwrite(prefix+"_gaussian.jpg", gaussian_image)
        cv2.imwrite(prefix+"_magnitude.jpg", sobel_image)
        cv2.imwrite(prefix+"_result.jpg", canny)


if __name__ == '__main__':
    main()
