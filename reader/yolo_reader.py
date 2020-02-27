import numpy as np
import cv2
from collections import OrderedDict
import copy
import functools
import paddle.fluid as fluid

from reader.insects_reader import get_insect_names, get_annotations, get_test_images
from process.image_ops import (DecodeImage, MixupImage, NormalizeBox, RandomDistort,
                               RandomRotateImage, RandomFlipImage, RandomCropImage,
                               RandomExpandImage, BboxXYXY2XYWH, RandomInterpImage,
                               NormalizeImage, Permute, BatchRandomReshape)


class YoloReader(object):

    def __init__(self, datadir,
                 sample_ops=[
                     DecodeImage(to_rgb=True, with_mixup=True),
                     MixupImage(alpha=1.5, beta=1.5),
                     RandomRotateImage(),
                     NormalizeBox(),
                     RandomDistort(),
                     RandomExpandImage(max_ratio=3, prob=0.5, keep_ratio=True,
                                       mean=[0.9076, 0.9265, 0.9232]),
                     RandomCropImage(),
                     RandomInterpImage(target_size=608),
                     RandomFlipImage(is_normalized=True),
                     NormalizeImage(mean=[0.9076, 0.9265, 0.9232], std=[0.1023, 0.1095, 0.1122],
                                    is_scale=True, is_channel_first=False),
                     BboxXYXY2XYWH(),
                     Permute(to_bgr=False)
                 ],
                 batch_ops=[
                     BatchRandomReshape(shapes=[320, 352, 384, 416, 448, 480,
                                                512, 544, 576, 608])
                 ],
                 test_image_shape=608,
                 num_max_boxes=50, shuffle_images=False, iters=10000,
                 mixup_epochs=200, batch_size=8, mode="train"):
        super(YoloReader, self).__init__()
        self.num_max_boxes = num_max_boxes
        self.mixup_epochs = mixup_epochs
        self.batch_size = batch_size
        self.sample_ops = sample_ops
        self.batch_ops = batch_ops
        self.datadir = datadir
        self.iters = iters
        self.mode = mode
        self.test_image_shape = test_image_shape
        self.shuffle_images = shuffle_images
        self._init_params()

    def get_bbox(self, sample):
        gt_bbox = sample["gt_bbox"]
        gt_class = sample["gt_class"]
        gt_score = sample["gt_score"]
        difficult = sample["difficult"]
        gt_bbox2 = np.zeros((self.num_max_boxes, 4))
        gt_class2 = np.zeros((self.num_max_boxes, ))
        gt_score2 = np.zeros((self.num_max_boxes, ))
        difficult2 = np.zeros((self.num_max_boxes, ))
        for i in range(len(gt_bbox)):
            gt_bbox2[i, :] = gt_bbox[i, :]
            gt_class2[i] = gt_class[i]
            gt_score2[i] = gt_score[i]
            difficult2[i] = difficult[i]
            if i >= self.num_max_boxes:
                break
        sample["gt_bbox"] = gt_bbox2
        sample["gt_class"] = gt_class2
        sample["gt_score"] = gt_score2
        sample["difficult"] = difficult2
        return sample

    def create_reader(self):
        self._load()
        size = len(self.records)

        def reader():
            _pos = 0
            batch_data = []
            total_pos = self.iters * self.batch_size if self.mode == "train" else size
            while _pos < total_pos:
                _ind = _pos % size
                if _ind == 0 and self.shuffle_images:
                    np.random.shuffle(self.records)
                _mix_ind = (_pos + self.mixup_epochs) % size
                batch_data.append((self.records[_ind], self.records[_mix_ind]))
                _pos += 1
                if len(batch_data) == self.batch_size:
                    yield batch_data
                    batch_data = []
            if len(batch_data) != 0:
                yield batch_data
                batch_data = []

        def get_data(items):
            batch_data = []
            for item in items:
                sample = item[0]
                mixup_sample = item[1]
                sample = self._next_sample(sample, mixup_sample)
                sample = self.get_img_data(sample)
                if self.mode != "test":
                    sample = self.get_bbox(sample)
                data = [sample[f] for f in self.fields]
                batch_data.append(data)
            for op in self.batch_ops:
                batch_data = op(batch_data)
            return self.make_array(batch_data)

        mapper = functools.partial(get_data, )

        return fluid.io.xmap_readers(mapper, reader, 16, 128)

    def _next_sample(self, record, mixup_record=None):
        sample = copy.deepcopy(record)
        if mixup_record:
            sample["mixup"] = copy.deepcopy(mixup_record)
        return sample

    def get_img_data(self, sample):
        for op in self.sample_ops:
            sample = op(sample)
        return sample

    def _load(self):
        cname2cid = get_insect_names()
        if self.mode == "train" or self.mode == "eval":
            self.records = get_annotations(cname2cid, self.datadir)
        else:
            self.records = get_test_images(self.datadir)

    def make_array(self, batch_data):
        data = {}
        for i, field in enumerate(self.fields):
            data[field] = np.array([item[i] for item in batch_data],
                                   dtype=self.field_types[field])
        return data

    def _init_params(self):
        self.field_types = {
            "image": "float32", "im_size": "int32", "im_id": "int64",
            "gt_bbox": "float32", "gt_class": "int32", "gt_score": "float32",
            "difficult": "int32", "objects": "int32"
        }
        if self.mode == "train":
            self.sample_ops = [
                DecodeImage(to_rgb=True, with_mixup=False),
                # MixupImage(alpha=1.2, beta=1.2),
                # RandomRotateImage(),
                NormalizeBox(),
                RandomDistort(),
                RandomExpandImage(max_ratio=3, prob=0.5, keep_ratio=True,
                                  mean=[0.9076, 0.9265, 0.9232]),
                RandomCropImage(),
                RandomInterpImage(target_size=608),
                # RandomFlipImage(is_normalized=True),
                NormalizeImage(mean=[0.9076, 0.9265, 0.9232], std=[0.1023, 0.1095, 0.1122],
                               is_scale=True, is_channel_first=False),
                BboxXYXY2XYWH(),
                Permute(to_bgr=False)
            ]
            self.batch_ops = [
                BatchRandomReshape(shapes=[320, 352, 384, 416, 448, 480,
                                           512, 544, 576, 608])
            ]
            self.fields = ["image", "gt_bbox", "gt_class", "gt_score"]
        elif self.mode == "eval":
            self.sample_ops = [
                DecodeImage(to_rgb=True, with_mixup=False),
                RandomInterpImage(target_size=self.test_image_shape, interp=cv2.INTER_CUBIC),
                NormalizeImage(mean=[0.9076, 0.9265, 0.9232], std=[0.1023, 0.1095, 0.1122],
                               is_scale=True, is_channel_first=False),
                Permute(to_bgr=False, channel_first=True)
            ]
            self.batch_ops = []
            self.fields = ["image", "im_size", "im_id", "gt_bbox",
                           "gt_class", "difficult", "objects"]
        elif self.mode == "test":
            self.sample_ops = [
                DecodeImage(to_rgb=True, with_mixup=False),
                RandomInterpImage(target_size=self.test_image_shape, interp=cv2.INTER_CUBIC),
                NormalizeImage(mean=[0.9076, 0.9265, 0.9232], std=[0.1023, 0.1095, 0.1122],
                               is_scale=True, is_channel_first=False),
                Permute(to_bgr=False)
            ]
            self.batch_ops = []
            self.fields = ["image", "im_size", "im_id"]

    def build_inputs(self, image_shape=[None, 3, None, None]):
        inputs_def = {
            'image': {'shape': image_shape, 'dtype': 'float32', 'lod_level': 0},
            'im_size': {'shape': [None, 2], 'dtype': 'int32', 'lod_level': 0},
            'im_id': {'shape': [None, 1], 'dtype': 'int64', 'lod_level': 0},
            'objects': {'shape': [None, 1], 'dtype': 'int32', 'lod_level': 0},
            'gt_bbox': {'shape': [None, self.num_max_boxes, 4], 'dtype': 'float32', 'lod_level': 0},
            'gt_class': {'shape': [None, self.num_max_boxes], 'dtype': 'int32', 'lod_level': 0},
            'gt_score': {'shape': [None, self.num_max_boxes], 'dtype': 'float32', 'lod_level': 0},
            'difficult': {'shape': [None, self.num_max_boxes], 'dtype': 'int32', 'lod_level': 0},
        }
        feed_vars = OrderedDict([(key, fluid.data(
            name=key,
            shape=inputs_def[key]['shape'],
            dtype=inputs_def[key]['dtype'],
            lod_level=inputs_def[key]['lod_level'])) for key in self.fields])
        return feed_vars
