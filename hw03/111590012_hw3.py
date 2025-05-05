import cv2
import os
import numpy as np
from heapq import heappush, heappop

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def create_result_dir():
    result_dir = os.path.join(CURRENT_DIR, 'result_img')
    os.makedirs(result_dir, exist_ok=True)


def get_all_test_img_path():
    test_dir = os.path.join(CURRENT_DIR, 'test_img')
    jpg_files = [os.path.join(test_dir, f)
                 for f in os.listdir(test_dir) if f.endswith('.png')]
    return jpg_files


def convert_to_gray(image):
    B, G, R = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    gray_image = (0.3*R + 0.59*G + 0.11*B).astype('uint8')
    return gray_image


drawing = False
current_label = 1
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0),
          (0, 255, 255), (255, 0, 255)]
color_table = {
    -1: (0, 0, 0),
    1: (0, 0, 255),
    2: (0, 255, 0),
    3: (255, 0, 0),
    4: (0, 255, 255),
    5: (255, 0, 255),
}
image = None
label_mask = None
label_map = None


def draw_label(event, x, y, flags, param):
    global drawing, current_label, label_mask, image

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        cv2.circle(image, (x, y), 5, colors[current_label - 1], -1)
        cv2.circle(label_mask, (x, y), 5, colors[current_label - 1], -1)
        cv2.circle(label_map, (x, y), 5, current_label, -1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.circle(image, (x, y), 5, colors[current_label - 1], -1)
            cv2.circle(label_mask, (x, y), 5, colors[current_label - 1], -1)
            cv2.circle(label_map, (x, y), 5, current_label, -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


def region_growing_segmentation(gray_image, label_map):
    h, w = gray_image.shape
    result = label_map.copy()
    queue = []

    def is_valid(x, y):
        return 0 <= x < h and 0 <= y < w

    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 將所有標記點的鄰居加入 queue
    for x in range(h):
        for y in range(w):
            if label_map[x, y] > 0:
                for dx, dy in neighbors:
                    nx, ny = x + dx, y + dy
                    if is_valid(nx, ny) and result[nx, ny] == 0:
                        result[nx, ny] = -2  # 加入 queue 的標記
                        heappush(queue, (gray_image[nx, ny], nx, ny))

    # 開始擴展
    while queue:
        _, x, y = heappop(queue)

        neighbor_labels = set()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and result[nx, ny] > 0:
                neighbor_labels.add(result[nx, ny])

        if len(neighbor_labels) == 1:
            result[x, y] = neighbor_labels.pop()
        elif len(neighbor_labels) >= 2:
            result[x, y] = -1  # 邊界
        else:
            continue

        # 將鄰近未標記點加入 queue
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if is_valid(nx, ny) and result[nx, ny] == 0:
                result[nx, ny] = -2
                heappush(queue, (gray_image[nx, ny], nx, ny))

    return result


def overlay_segmentation_on_image(original_image, segmented_image, alpha=0.5):
    h, w = segmented_image.shape
    segmentation_color = np.zeros((h, w, 3), dtype=np.uint8)

    for label, color in color_table.items():
        mask = (segmented_image == label)
        segmentation_color[mask] = color

    original_float = original_image.astype(np.float32)
    segment_float = segmentation_color.astype(np.float32)

    result = (alpha * segment_float + (1 - alpha)
              * original_float).astype(np.uint8)

    return result


def run_manual_labeling():
    global image, current_label
    while True:
        cv2.imshow("Mark Labels", image)
        key = cv2.waitKey(1) & 0xFF

        if key in [ord(str(k)) for k in range(1, 6)]:
            current_label = int(chr(key))
            print(f"切換至標籤 {current_label}")
        elif key == ord('q'):
            break


def main():
    global image, label_mask, label_map, current_label
    cv2.namedWindow('Mark Labels')
    cv2.setMouseCallback('Mark Labels', draw_label)

    create_result_dir()
    image_paths = get_all_test_img_path()

    for image_path in image_paths:
        origin_image = cv2.imread(image_path)

        image = origin_image.copy()
        h, w = image.shape[:2]
        label_mask = np.zeros((h, w, 3), dtype=np.uint8)
        label_map = np.zeros((h, w), dtype=np.int32)

        run_manual_labeling()

        gray_image = convert_to_gray(origin_image)
        segmented_image = region_growing_segmentation(gray_image, label_map)
        segmented_image_with_color = overlay_segmentation_on_image(
            origin_image, segmented_image)

        # save result
        prefix = os.path.join(CURRENT_DIR, 'result_img',
                              os.path.splitext(os.path.basename(image_path))[0])
        cv2.imwrite(prefix+"_marked.jpg", image)
        cv2.imwrite(prefix+"_mask.jpg", label_mask)
        cv2.imwrite(prefix+"_seg.jpg", segmented_image_with_color)


if __name__ == '__main__':
    main()
