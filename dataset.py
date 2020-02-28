import os
import sys
import re
import six
import math
from natsort import natsorted
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow import data


# import torch
# from torch.utils.data import Dataset, ConcatDataset, Subset
# from torch._utils import _accumulate
# import torchvision.transforms as transforms


def _read_img_paths_and_labels(annotation_paths):
    """读取annotation文件，获取图片路径与标签"""
    img_paths = []
    labels = []
    for annotation_path in annotation_paths.split(','):
        # 分隔符为',' 可以自行修改
        # annotation_path的目录下，有annotation文件以及图片
        annotation_folder = os.path.dirname(annotation_path)
        with open(annotation_path) as f:
            content = np.array(
                [line.strip().split(',') for line in f.readlines()])
        #     lines = f.readlines()
        # part_img_paths = [line.strip().split(',')[0] for line in lines]
        # part_labels = [line.strip().split(',')[1] for line in lines]

        part_img_paths = content[:, 0]

        # Parse MjSynth dataset. format: XX_label_XX.jpg XX
        # URL: https://www.robots.ox.ac.uk/~vgg/data/text/
        # part_labels = [line.split("_")[1] for line in part_img_paths]

        # Parse example dataset. format: XX.jpg label
        part_labels = content[:, 1]

        part_img_paths = [os.path.join(annotation_folder, line)
                          for line in part_img_paths]
        img_paths.extend(part_img_paths)
        labels.extend(part_labels)

    return img_paths, labels


class OCR_DataLoader(object):
    def __init__(self,
                 annotation_paths,
                 image_height,
                 image_width,
                 table_path,
                 blank_index=0,
                 batch_size=1,
                 shuffle=False,
                 rgb=False,
                 keep_ratio_with_pad=False,
                 repeat=1):

        img_paths, labels = _read_img_paths_and_labels(annotation_paths)
        self.batch_size = batch_size
        self.image_width = image_width
        self.image_height = image_height
        self.size = len(img_paths)
        self.rgb = rgb
        self.pad = keep_ratio_with_pad

        file_init = tf.lookup.TextFileInitializer(
            table_path,
            tf.string,
            tf.lookup.TextFileIndex.WHOLE_LINE,
            tf.int64,
            tf.lookup.TextFileIndex.LINE_NUMBER)
        # default_value: The value to use if a key is missing in the table.
        self.table = tf.lookup.StaticHashTable(
            initializer=file_init,
            default_value=blank_index)

        dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
        # print(dataset.element_spec)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=self.size, seed=6657)
        dataset = dataset.map(self._decode_and_resize)
        # Experimental function.
        # Ignore the errors e.g. decode error or invalid data.
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(self._convert_label)
        dataset = dataset.repeat(repeat)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        self.dataset = dataset

    def _decode_and_resize(self, filename, label):
        image = tf.io.read_file(filename)
        # _, ext = os.path.splitext(filename)
        _, ext = None, '.png'
        ext = ext.lower()
        channels = 1
        if self.rgb:
            channels = 3
        if ext == '.jpg' or ext == '.jpeg':
            image = tf.io.decode_jpeg(image, channels=channels)
        if ext == '.png':
            image = tf.io.decode_png(image, channels=channels)
        image = tf.image.convert_image_dtype(image, tf.float32)
        if self.pad:
            image = tf.image.resize_with_pad(image, self.image_height, self.image_width, method='bicubic')
        else:
            image = tf.image.resize(image, [self.image_height, self.image_width], method='bicubic')
        return image, label

    def _convert_label(self, image, label):
        # According to official document, only dense tensor will run on GPU
        # or TPU, but I have tried convert label to dense tensor by `to_tensor`
        # and `row_lengths`, the speed of training step slower than sparse.
        chars = tf.strings.unicode_split(label, input_encoding="UTF-8")
        mapped_label = tf.ragged.map_flat_values(self.table.lookup, chars)
        sparse_label = mapped_label.to_sparse()
        sparse_label = tf.cast(sparse_label, tf.int32)
        return image, sparse_label

    def __call__(self):
        """Return tf.data.Dataset."""
        return self.dataset

    def __len__(self):
        return self.size


class MyRawDataset(data.Dataset):

    def __init__(self, img_list, opt):
        self.opt = opt
        self.image_path_list = img_list
        # self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def map_to_chars(inputs, table, blank_index=0, merge_repeated=False):
    """Map to chars.

    Args:
        inputs: list of char ids.
        table: char map.
        blank_index: the index of blank.
        merge_repeated: True, Only if tf decoder is not used.
    Returns:
        lines: list of string.
    """
    lines = []
    for line in inputs:
        text = ""
        previous_char = -1
        for char_index in line:
            if merge_repeated:
                if char_index == previous_char:
                    continue
            previous_char = char_index
            if char_index == blank_index:
                continue
            text += table[char_index]
        lines.append(text)
    return lines


def map_and_count(decoded, Y, mapper, blank_index=0, merge_repeated=False):
    decoded = tf.sparse.to_dense(decoded[0], default_value=blank_index).numpy()
    decoded = map_to_chars(decoded, mapper, blank_index=blank_index,
                           merge_repeated=merge_repeated)
    Y = tf.sparse.to_dense(Y, default_value=blank_index).numpy()
    Y = map_to_chars(Y, mapper, blank_index=blank_index,
                     merge_repeated=merge_repeated)
    cnt = 0
    for y_pred, y in zip(decoded, Y):
        if y_pred == y:
            cnt += 1
    return cnt