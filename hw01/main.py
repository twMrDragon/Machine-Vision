import cv2
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

def create_result_dir():
    result_dir = os.path.join(CURRENT_DIR, 'result_img')
    os.makedirs(result_dir, exist_ok=True)

def get_all_test_img_path():
    test_dir = os.path.join(CURRENT_DIR, 'test_img')
    jpg_files = [os.path.join(test_dir,f) for f in os.listdir(test_dir) if f.endswith('.jpg')]
    return jpg_files

def convert_to_gray(image):
    B,G,R = image[:,:,0], image[:,:,1], image[:,:,2]
    gray_image = (0.3*R + 0.59*G + 0.11*B).astype('uint8')
    return  gray_image

def q11():
    image_paths = get_all_test_img_path()
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray_image = convert_to_gray(image)
        result_path = os.path.join(CURRENT_DIR, 'result_img', os.path.splitext(os.path.basename(image_path))[0]+"_q1-1.jpg")
        cv2.imwrite(result_path, gray_image)

def convert_to_binary(gray_image, threshold):
    binary_image = gray_image.copy()
    binary_image[binary_image<threshold] = 0
    binary_image[binary_image>=threshold] = 255
    return binary_image

def q12():
    image_paths = get_all_test_img_path()
    for image_path in image_paths:
        image = cv2.imread(image_path)
        gray_image = convert_to_gray(image)
        binary_image = convert_to_binary(gray_image, 128)
        result_path = os.path.join(CURRENT_DIR, 'result_img', os.path.splitext(os.path.basename(image_path))[0]+"_q1-2.jpg")
        cv2.imwrite(result_path, binary_image)


def convert_to_index_color(image):
    pass

def q13():
    pass

def q1():
    q11()
    q12()
    q13()

def main():
    create_result_dir()
    q1()

if __name__ == '__main__':
    main()