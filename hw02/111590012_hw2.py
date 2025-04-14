import cv2
import os
import numpy as np
from enum import Enum


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


def convert_to_binary(gray_image, threshold, flip=False):
    binary_image = gray_image.copy()
    smaller = binary_image < threshold
    larger = binary_image >= threshold
    binary_image[smaller] = (0 if not flip else 255)
    binary_image[larger] = (255 if not flip else 0)
    return binary_image


def iterative_threshold(image, epsilon=1):
    T = np.mean(image)
    while True:
        G1 = image[image > T]
        G2 = image[image <= T]
        new_T = (np.mean(G1) + np.mean(G2)) / 2
        if abs(new_T - T) < epsilon:
            break
        T = new_T
    return T


class Connected(Enum):
    FOUR = 1
    ENIGHT = 2


def find_root(label_equiv, label):
    root = label
    while label_equiv[root] != root:
        root = label_equiv[root]
    while label != root:
        parent = label_equiv[label]
        label_equiv[label] = root
        label = parent
    return root


def component_labeling(binary_image, conn=Connected.FOUR):
    h, w = binary_image.shape
    labels = np.zeros((h, w), dtype=np.int32)
    label = 1
    label_equiv = {}

    check_list = [
        (0, -1), (-1, 0)] if conn == Connected.FOUR else [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

    for i in range(h):
        for j in range(w):
            if binary_image[i, j] == 0:
                continue
            neighbors = []

            for di, dj in check_list:
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and labels[ni, nj] > 0:
                    neighbors.append(labels[ni, nj])

            if not neighbors:
                labels[i, j] = label
                label_equiv[label] = label
                label += 1
            else:
                min_label = min(neighbors)
                labels[i, j] = min_label
                for neighbor_label in neighbors:
                    root1 = find_root(label_equiv, neighbor_label)
                    root2 = find_root(label_equiv, min_label)
                    if root1 != root2:
                        label_equiv[root2] = root1

    for i in range(h):
        for j in range(w):
            if labels[i, j] > 0:
                labels[i, j] = find_root(label_equiv, labels[i, j])

    return labels


def color_labeling(labels):
    h, w = labels.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)

    unique_labels = np.unique(labels)
    np.random.seed(42)
    colors = {label: np.random.randint(
        0, 255, size=3, dtype=np.uint8) for label in unique_labels}

    # 背景的顏色設定為黑色
    colors[0] = [0, 0, 0]

    for label, color in colors.items():
        color_img[labels == label] = color

    return color_img


def pattern_counting(binary_image, pattern):
    count = 0
    for i in range(binary_image.shape[0] - 1):
        for j in range(binary_image.shape[1] - 1):
            if np.array_equal(binary_image[i:i+2, j:j+2], pattern):
                count += 1
    return count


def internal_counting(binary_image):
    internal = 0
    for i in range(2):
        for j in range(2):
            pattern = np.full((2, 2), 255, dtype=int)
            pattern[i][j] = 0
            internal += pattern_counting(binary_image, pattern)
    return internal


def external_counting(binary_image):
    external = 0
    for i in range(2):
        for j in range(2):
            pattern = np.zeros((2, 2), dtype=int)
            pattern[i][j] = 255
            external += pattern_counting(binary_image, pattern)
    return external


def object_counting(binary_image):
    internal_count = internal_counting(binary_image)
    external_count = external_counting(binary_image)
    object_count = (external_count-internal_count)/4
    return internal_count, external_count, object_count


def main():
    create_result_dir()
    image_paths = get_all_test_img_path()
    for image_path in image_paths:
        origin_image = cv2.imread(image_path)
        gray_image = convert_to_gray(origin_image)
        threshold = iterative_threshold(gray_image)
        # threshold = 250
        binary_image = convert_to_binary(gray_image, threshold, flip=True)

        labels = component_labeling(
            binary_image, Connected.FOUR)
        color_labels_4 = color_labeling(labels)
        labels = component_labeling(
            binary_image, Connected.ENIGHT)
        color_labels_8 = color_labeling(labels)

        internal_count, external_count, object_count = object_counting(
            binary_image)

        print(os.path.basename(image_path).center(30, "="))
        print(f"Internal Count: {internal_count}")
        print(f"External Count: {external_count}")
        print(f"Object Count: {object_count}")

        # save result
        prefix = os.path.join(CURRENT_DIR, 'result_img', os.path.splitext(
            os.path.basename(image_path))[0])
        cv2.imwrite(prefix+"_4.jpg", color_labels_4)
        cv2.imwrite(prefix+"_8.jpg", color_labels_8)


if __name__ == '__main__':
    main()
