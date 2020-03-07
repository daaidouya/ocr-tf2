import cv2
import math
from PIL import Image, ImageDraw
import os
import numpy as np


def extract_vertices_labels(lines):
    labels = []
    vertices = []
    for line in lines:
        line = line.rstrip('\n').lstrip('\ufeff').split(';')
        vertices.append(list(map(int, line[1:9])))
        label = line[-1]
        labels.append(label)
    return np.array(vertices), np.array(labels)


def plot_boxes(img, boxes):
    if boxes is None:
        return img
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.polygon([box[0], box[1], box[2], box[3],
                      box[4], box[5], box[6], box[7]], outline=(0, 255, 0))
    return img


def crop_img_v2(save_dir, img_path, image_name, pos, no, is_right, w, h, label):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    img = cv2.imread(img_path, 0)
    # TODO：检测出来的坐标需要为原图坐标
    # if not is_right:
    #     img = cv2.flip(img, -1)

    output_w = int(round(w, 0))
    output_h = int(round(h, 0))
    src_poly = np.array(pos).reshape((4, 2)).astype(np.float32)
    dst_poly = np.array([[0, 0], [output_w, 0],
                         [output_w, output_h], [0, output_h]]).astype(np.float32)
    # 使用透视变换
    M = cv2.getPerspectiveTransform(src_poly, dst_poly)
    result = cv2.warpPerspective(img, M, (output_w, output_h))

    return result


def main_func_test(temp):
    base_path = r'C:\Users\chd19\Desktop\validate\\'
    imgs_path = base_path + temp + '\\Image'
    labels_path = base_path + temp + '\\Label'
    out_path = r'C:\Users\chd19\Desktop\crop\rmb_money_val\\'

    img_names = sorted(os.listdir(imgs_path))[1200:]
    img_files = [os.path.join(imgs_path, img_file)
                 for img_file in sorted(os.listdir(imgs_path))]
    label_files = [os.path.join(labels_path, label_file)
                   for label_file in sorted(os.listdir(labels_path))]
    f1 = open(r'C:\Users\chd19\Desktop\crop\val_annotations.txt', 'w+')
    for img_num, img_name in enumerate(img_names):
        # print(boxes)
        with open(label_files[img_num], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        vertices, labels = extract_vertices_labels(lines)

        # 判断图片是否需要180度翻转
        is_right = True if vertices[0][1] < vertices[3][1] else False

        # save_dir = out_path + img_name.split(".")[0]
        # os.mkdir(save_dir)
        save_dir = out_path
        # print(boxes)
        for no, vertice in enumerate(vertices[3:4]):
            x1, y1, x2, y2, x3, y3, x4, y4 = vertice
            w = max(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2),
                    math.sqrt((x4 - x3) ** 2 + (y4 - y3) ** 2))
            h = max(math.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2),
                    math.sqrt((x4 - x1) ** 2 + (y4 - y1) ** 2))
            result = crop_img_v2(save_dir, img_files[img_num], img_name,
                        vertice, no, is_right, w, h, labels[no + 3])
            save_name = img_name.split(".")[0] + '_' + str(no) + '.jpg'
            cv2.imwrite(os.path.join(save_dir, save_name), result)
            f1.write('rmb_money_val/' + save_name + ' ' + labels[no + 3] + '\n')
        print("crop image " + str(img_num) + " " + img_name + " finished.")


if __name__ == '__main__':
    main_func_test('t1')
