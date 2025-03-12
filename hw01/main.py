import cv2
import os
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

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


def convert_to_binary(gray_image, threshold):
    binary_image = gray_image.copy()
    binary_image[binary_image < threshold] = 0
    binary_image[binary_image >= threshold] = 255
    return binary_image


def convert_to_index_color(image):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    # kmeans 分 32 組找出 32 個顏色
    kmeans = KMeans(n_clusters=32, n_init=10, random_state=42)
    kmeans.fit(pixels)
    color_map = kmeans.cluster_centers_.astype(np.uint8)
    # 用 KDTree 找出每個 pixel 最近的顏色
    kdtree = KDTree(color_map)
    _, indices = kdtree.query(pixels)
    # 重組成圖片
    indexed_image = indices.reshape(h, w)
    indexed_colored_image = color_map[indexed_image]
    return indexed_colored_image


def resize_image_without_interpolation(image, scale):
    h, w, _ = image.shape
    resize_h, resize_w = int(h*scale), int(w*scale)
    resized_image = np.zeros((resize_h, resize_w, 3), dtype=image.dtype)
    for i in range(resize_h):
        for j in range(resize_w):
            resized_image[i, j] = image[int(i/scale), int(j/scale)]
    return resized_image


def resize_image_with_interpolation(image, scale):
    h, w, __doc__ = image.shape
    resize_h, resize_w = int(h*scale), int(w*scale)
    resized_image = np.zeros((resize_h, resize_w, 3), dtype=image.dtype)
    for y in range(resize_h):
        for x in range(resize_w):
            # 找到對應的原始座標 (非整數)
            src_x = x / scale
            src_y = y / scale

            # 計算四個最近的整數座標
            x1, y1 = int(np.floor(src_x)), int(np.floor(src_y))
            x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

            # 計算距離比例
            dx, dy = src_x - x1, src_y - y1

            Q11 = image[y1, x1].astype(np.float32)
            Q21 = image[y1, x2].astype(np.float32)
            Q12 = image[y2, x1].astype(np.float32)
            Q22 = image[y2, x2].astype(np.float32)
            R1 = (1 - dx) * Q11 + dx * Q21
            R2 = (1 - dx) * Q12 + dx * Q22

            P = (1 - dy) * R1 + dy * R2

            resized_image[y, x] = P.astype(np.uint8)

    return resized_image


def main():
    create_result_dir()
    image_paths = get_all_test_img_path()
    for image_path in image_paths:
        origin_image = cv2.imread(image_path)
        # q1
        gray_image = convert_to_gray(origin_image)
        binary_image = convert_to_binary(gray_image, 128)
        index_color_image = convert_to_index_color(origin_image)
        # q2
        half_image = resize_image_without_interpolation(origin_image, 0.5)
        double_image = resize_image_without_interpolation(origin_image, 2)
        half_image_interpolation = resize_image_with_interpolation(
            origin_image, 0.5)
        double_image_interpolation = resize_image_with_interpolation(
            origin_image, 2)
        # save result
        prefix = os.path.join(CURRENT_DIR, 'result_img', os.path.splitext(
            os.path.basename(image_path))[0])
        cv2.imwrite(prefix+"_q1-1.jpg", gray_image)
        cv2.imwrite(prefix+"_q1-2.jpg", binary_image)
        cv2.imwrite(prefix+"_q1-3.jpg", index_color_image)
        cv2.imwrite(prefix+"_q2-1_half.jpg", half_image)
        cv2.imwrite(prefix+"_q2-1_double.jpg", double_image)
        cv2.imwrite(prefix+"_q2-2_half.jpg", half_image_interpolation)
        cv2.imwrite(prefix+"_q2-2_double.jpg", double_image_interpolation)


if __name__ == '__main__':
    main()
