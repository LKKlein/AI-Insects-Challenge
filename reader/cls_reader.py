import os
import math
import random
import functools
import numpy as np
import cv2
import logging

from paddle import fluid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

random.seed(0)
np.random.seed(0)


def random_crop(img, size, settings, scale=None, ratio=None,
                interpolation=None):
    lower_scale = settings["lower_scale"]
    lower_ratio = settings["lower_ratio"]
    upper_ratio = settings["upper_ratio"]
    scale = [lower_scale, 1.0] if scale is None else scale
    ratio = [lower_ratio, upper_ratio] if ratio is None else ratio

    aspect_ratio = math.sqrt(np.random.uniform(*ratio))
    w = 1. * aspect_ratio
    h = 1. / aspect_ratio

    bound = min((float(img.shape[0]) / img.shape[1]) / (h**2),
                (float(img.shape[1]) / img.shape[0]) / (w**2))

    scale_max = min(scale[1], bound)
    scale_min = min(scale[0], bound)

    target_area = img.shape[0] * img.shape[1] * np.random.uniform(scale_min,
                                                                  scale_max)
    target_size = math.sqrt(target_area)
    w = int(target_size * w)
    h = int(target_size * h)
    i = np.random.randint(0, img.shape[0] - h + 1)
    j = np.random.randint(0, img.shape[1] - w + 1)

    img = img[i:i + h, j:j + w, :]

    if interpolation:
        resized = cv2.resize(img, (size, size), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (size, size))
    return resized


def resize_short(img, target_size, interpolation=None):
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    if interpolation:
        resized = cv2.resize(
            img, (resized_width, resized_height), interpolation=interpolation)
    else:
        resized = cv2.resize(img, (resized_width, resized_height))
    return resized


def crop_image(img, target_size, center):
    height, width = img.shape[:2]
    size = target_size
    if center:
        w_start = (width - size) // 2
        h_start = (height - size) // 2
    else:
        w_start = np.random.randint(0, width - size + 1)
        h_start = np.random.randint(0, height - size + 1)
    w_end = w_start + size
    h_end = h_start + size
    img = img[h_start:h_end, w_start:w_end, :]
    return img


def process_image(sample, settings, mode):
    mean = settings["image_mean"]
    std = settings["image_std"]
    crop_size = settings["image_shape"][1]

    if "image" not in sample and "im_file" in sample:
        img_path = sample["im_file"]
        img = cv2.imread(img_path)
    else:
        img = sample["image"]

    if mode == 'train':
        if crop_size > 0:
            img = random_crop(
                img, crop_size, settings, interpolation=settings["interpolation"])
        if np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
    else:
        if crop_size > 0:
            target_size = settings["resize_short_size"]
            img = resize_short(
                img, target_size, interpolation=settings["interpolation"])
            img = crop_image(img, target_size=crop_size, center=True)

    img = img[:, :, ::-1]

    img = img.astype('float32').transpose((2, 0, 1)) / 255
    img_mean = np.array(mean).reshape((3, 1, 1))
    img_std = np.array(std).reshape((3, 1, 1))
    img -= img_mean
    img /= img_std
    # doing training
    if mode == 'train' or mode == 'val':
        return (img, sample["label"])
    # doing predict
    elif mode == 'test':
        return (img, )


def process_batch_data(samples, settings, mode):
    batch_data = []
    for sample in samples:
        tmp_data = process_image(sample, settings, mode)
        if tmp_data is None:
            continue
        batch_data.append(tmp_data)
    return batch_data


class DataReader:

    def __init__(self, batch_size=64, seed=None):
        self.shuffle_seed = seed
        self.batch_size = batch_size

    def set_shuffle_seed(self, seed):
        self.shuffle_seed = seed

    def _reader_creator(self, settings, records, mode, shuffle=False):

        def reader():
            def read_sample_list():
                if mode == "train" and shuffle:
                    np.random.RandomState(self.shuffle_seed).shuffle(records)

                batch_data = []
                for record in records:
                    batch_data.append(record)
                    if len(batch_data) == self.batch_size:
                        yield batch_data
                        batch_data = []

                if len(batch_data) != 0:
                    yield batch_data
                    batch_data = []

            return read_sample_list

        data_reader = reader()

        mapper = functools.partial(process_batch_data, settings=settings, mode=mode)

        return fluid.io.xmap_readers(mapper, data_reader, 8, 2048, order=False)

    def train(self, settings):
        file_list = os.path.join(settings["data_dir"], 'train_list.txt')
        with open(file_list) as f:
            records = [line.strip().split() for line in f]
            records = [{"im_file": record[0], "label": int(record[1])}
                       for record in records]

        reader = self._reader_creator(settings, records, 'train', shuffle=True)
        return reader

    def val(self, settings):
        file_list = os.path.join(settings["data_dir"], 'val_list.txt')
        with open(file_list) as f:
            records = [line.strip().split() for line in f]
            records = [{"im_file": record[0], "label": int(record[1])}
                       for record in records]

        return self._reader_creator(settings, records, 'val', shuffle=False)

    def test(self, settings):
        images = settings["images"]
        img_count = settings["images_num"]
        records = [{"image": images[i]} for i in range(img_count)]
        return self._reader_creator(settings, records, 'test', shuffle=False)
