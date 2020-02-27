import numpy as np
import cv2
import math
from PIL import Image, ImageEnhance
import random

from utils.box_utils import (multi_box_iou, box_crop, filter_and_process,
                             generate_sample_bbox, satisfy_sample_constraint, clip_bbox)


class DecodeImage(object):

    def __init__(self, to_rgb=True, with_mixup=False):
        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup

    def __call__(self, sample):
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        if "im_size" not in sample:
            sample["im_size"] = np.array([im.shape[0], im.shape[1]])

        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.int32)
        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
            self.__call__(sample['mixup'])
        return sample


class RandomRotateImage(object):

    def __init__(self, max_angle=90, scale=1., mean=[0.9076, 0.9265, 0.9232]):
        self.max_angle = max_angle
        self.angles = list(range(-95, -85)) + list(range(-5, 5)) + list(range(85, 95))
        self.scale = scale
        self.mean = np.array(mean) * 255
        self.mean = self.mean.astype("int").tolist()

    def __call__(self, sample):
        img = sample["image"]
        bboxes = sample["gt_bbox"]
        w = sample['w']
        h = sample['h']
        angle = np.random.choice(self.angles)

        cx, cy = w // 2, h // 2
        M = cv2.getRotationMatrix2D((cx, cy), angle, self.scale)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        nW = h * sin + w * cos
        nH = h * cos + w * sin

        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        rot_img = cv2.warpAffine(img, M, (int(nW), int(nH)), borderValue=self.mean)

        shift = int((nW - w) / 2)
        rot_img = rot_img[shift: int(nH - shift), shift: int(nW - shift)]

        rot_bboxes = np.zeros_like(bboxes)
        for idx, box in enumerate(bboxes):
            new_box = [[box[0], box[1]], [box[2], box[1]],
                       [box[0], box[3]], [box[2], box[3]]]
            out_box = [nW, nH, 0, 0]
            for j, coord in enumerate(new_box):
                v = [coord[0], coord[1], 1]
                calculated = np.dot(M, v)
                out_box[0] = min(out_box[0], calculated[0])
                out_box[1] = min(out_box[1], calculated[1])
                out_box[2] = max(out_box[2], calculated[0])
                out_box[3] = max(out_box[3], calculated[1])
            rot_bboxes[idx] = np.array(out_box) - shift
        sample["image"] = rot_img
        sample["gt_bbox"] = rot_bboxes
        sample["h"] = rot_img.shape[0]
        sample["w"] = rot_img.shape[1]
        return sample


class RandomFlipImage(object):

    def __init__(self, prob=0.5, is_normalized=False):
        self.prob = prob
        self.is_normalized = is_normalized

    def __call__(self, sample):
        gt_bbox = sample['gt_bbox']
        im = sample['image']
        height, width, _ = im.shape
        if np.random.uniform(0, 1) < self.prob:
            im = im[:, ::-1, :]
            if gt_bbox.shape[0] == 0:
                return sample
            oldx1 = gt_bbox[:, 0].copy()
            oldx2 = gt_bbox[:, 2].copy()
            if self.is_normalized:
                gt_bbox[:, 0] = 1 - oldx2
                gt_bbox[:, 2] = 1 - oldx1
            else:
                gt_bbox[:, 0] = width - oldx2 - 1
                gt_bbox[:, 2] = width - oldx1 - 1
            sample['gt_bbox'] = gt_bbox
            sample['image'] = im
        if np.random.uniform(0, 1) < self.prob:
            im = im[::-1, :, :]
            if gt_bbox.shape[0] == 0:
                return sample
            oldy1 = gt_bbox[:, 1].copy()
            oldy2 = gt_bbox[:, 3].copy()
            if self.is_normalized:
                gt_bbox[:, 1] = 1 - oldy2
                gt_bbox[:, 3] = 1 - oldy1
            else:
                gt_bbox[:, 1] = height - oldy2 - 1
                gt_bbox[:, 3] = height - oldy1 - 1
            sample['gt_bbox'] = gt_bbox
            sample['image'] = im
        return sample


class NormalizeImage(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[1, 1, 1],
                 is_scale=True, is_channel_first=True):
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, sample):
        im = sample["image"]
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        sample["image"] = im
        return sample


class RandomDistort(object):

    def __init__(self, brightness_range=[0.5, 1.5], contrast_range=[0.5, 1.5],
                 saturation_range=[0.5, 1.5], hue_range=[-18, 18], prob=0.5,
                 count=4, is_order=False):
        super(RandomDistort, self).__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.hue_range = hue_range
        self.prob = prob
        self.count = count
        self.is_order = is_order

    def random_brightness(self, img):
        brightness_delta = np.random.uniform(self.brightness_range[0],
                                             self.brightness_range[1])
        prob = np.random.uniform(0, 1)
        if prob < self.prob:
            img = ImageEnhance.Brightness(img).enhance(brightness_delta)
        return img

    def random_contrast(self, img):
        contrast_delta = np.random.uniform(self.contrast_range[0],
                                           self.contrast_range[1])
        prob = np.random.uniform(0, 1)
        if prob < self.prob:
            img = ImageEnhance.Contrast(img).enhance(contrast_delta)
        return img

    def random_saturation(self, img):
        saturation_delta = np.random.uniform(self.saturation_range[0],
                                             self.saturation_range[1])
        prob = np.random.uniform(0, 1)
        if prob < self.prob:
            img = ImageEnhance.Color(img).enhance(saturation_delta)
        return img

    def random_hue(self, img):
        hue_delta = np.random.uniform(self.hue_range[0], self.hue_range[1])
        prob = np.random.uniform(0, 1)
        if prob < self.prob:
            img = np.array(img.convert('HSV'))
            img[:, :, 0] = img[:, :, 0] + hue_delta
            img = Image.fromarray(img, mode='HSV').convert('RGB')
        return img

    def __call__(self, sample):
        ops = [
            self.random_brightness, self.random_contrast,
            self.random_saturation, self.random_hue
        ]
        if self.is_order:
            prob = np.random.uniform(0, 1)
            if prob < 0.5:
                ops = [
                    self.random_brightness,
                    self.random_saturation,
                    self.random_hue,
                    self.random_contrast,
                ]
        else:
            ops = random.sample(ops, self.count)
        im = sample['image']
        im = Image.fromarray(im)
        for id in range(self.count):
            im = ops[id](im)
        im = np.asarray(im)
        sample['image'] = im
        return sample


class ExpandImage(object):

    def __init__(self, max_ratio, prob, mean=[127.5, 127.5, 127.5]):
        super(ExpandImage, self).__init__()
        self.max_ratio = max_ratio
        self.mean = mean
        self.prob = prob

    def __call__(self, sample):
        prob = np.random.uniform(0, 1)
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']
        if prob < self.prob:
            if self.max_ratio - 1 >= 0.01:
                expand_ratio = np.random.uniform(1, self.max_ratio)
                height = int(im_height * expand_ratio)
                width = int(im_width * expand_ratio)
                h_off = math.floor(np.random.uniform(0, height - im_height))
                w_off = math.floor(np.random.uniform(0, width - im_width))
                expand_bbox = [
                    -w_off / im_width, -h_off / im_height,
                    (width - w_off) / im_width, (height - h_off) / im_height
                ]
                expand_im = np.ones((height, width, 3))
                expand_im = np.uint8(expand_im * np.squeeze(self.mean))
                expand_im = Image.fromarray(expand_im)
                im = Image.fromarray(im)
                expand_im.paste(im, (int(w_off), int(h_off)))
                expand_im = np.asarray(expand_im)
                gt_bbox, gt_class, _ = filter_and_process(expand_bbox, gt_bbox,
                                                          gt_class)
                sample['image'] = expand_im
                sample['gt_bbox'] = gt_bbox
                sample['gt_class'] = gt_class
                sample['w'] = width
                sample['h'] = height

        return sample


class RandomExpandImage(object):

    def __init__(self, max_ratio, prob, mean=[0.9076, 0.9265, 0.9232], keep_ratio=True):
        super(RandomExpandImage, self).__init__()
        self.max_ratio = max_ratio
        self.keep_ratio = keep_ratio
        self.mean = mean
        self.prob = prob

    def __call__(self, sample):
        if np.random.uniform(0., 1.) < self.prob:
            return sample

        img = sample['image']
        im_height = int(sample['h'])
        im_width = int(sample['w'])
        gt_bbox = sample['gt_bbox']

        ratio_x = np.random.uniform(1., self.max_ratio)
        if self.keep_ratio:
            ratio_y = ratio_x
        else:
            ratio_y = np.random.uniform(1., self.max_ratio)

        oh = int(im_height * ratio_y)
        ow = int(im_width * ratio_x)
        if oh <= im_height or ow <= im_width:
            return sample

        off_x = np.random.randint(0, ow - im_width)
        off_y = np.random.randint(0, oh - im_height)

        channel = 3
        out_img = np.zeros((oh, ow, channel))
        if self.mean and len(self.mean) == channel:
            for i in range(channel):
                out_img[:, :, i] = self.mean[i] * 255.0

        out_img[off_y:off_y + im_height, off_x:off_x + im_width, :] = img.astype(np.uint8)
        gt_bbox[:, 0] = ((gt_bbox[:, 0] * im_width) + off_x) / float(ow)
        gt_bbox[:, 1] = ((gt_bbox[:, 1] * im_height) + off_y) / float(oh)
        gt_bbox[:, 2] = ((gt_bbox[:, 2] * im_width) + off_x) / float(ow)
        gt_bbox[:, 3] = ((gt_bbox[:, 3] * im_height) + off_y) / float(oh)

        sample['w'] = ow
        sample['h'] = oh
        sample['image'] = out_img.astype('uint8')
        sample['gt_bbox'] = gt_bbox
        return sample


class CropImage(object):

    def __init__(self, batch_sampler, satisfy_all=False, avoid_no_bbox=True):
        super(CropImage, self).__init__()
        self.batch_sampler = batch_sampler
        self.satisfy_all = satisfy_all
        self.avoid_no_bbox = avoid_no_bbox

    def __call__(self, sample, context):
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']
        gt_score = sample['gt_score']
        sampled_bbox = []
        gt_bbox = gt_bbox.tolist()
        for sampler in self.batch_sampler:
            found = 0
            for i in range(sampler[1]):
                if found >= sampler[0]:
                    break
                sample_bbox = generate_sample_bbox(sampler)
                if satisfy_sample_constraint(sampler, sample_bbox, gt_bbox,
                                             self.satisfy_all):
                    sampled_bbox.append(sample_bbox)
                    found = found + 1
        im = np.array(im)
        while sampled_bbox:
            idx = int(np.random.uniform(0, len(sampled_bbox)))
            sample_bbox = sampled_bbox.pop(idx)
            sample_bbox = clip_bbox(sample_bbox)
            crop_bbox, crop_class, crop_score = \
                filter_and_process(sample_bbox, gt_bbox, gt_class, gt_score)
            if self.avoid_no_bbox:
                if len(crop_bbox) < 1:
                    continue
            xmin = int(sample_bbox[0] * im_width)
            xmax = int(sample_bbox[2] * im_width)
            ymin = int(sample_bbox[1] * im_height)
            ymax = int(sample_bbox[3] * im_height)
            im = im[ymin:ymax, xmin:xmax]
            sample['image'] = im
            sample['gt_bbox'] = crop_bbox
            sample['gt_class'] = crop_class
            sample['gt_score'] = crop_score
            return sample
        return sample


class RandomCropImage(object):

    def __init__(self, scales=[0.3, 1.0], max_ratio=2.0, max_trial=50,
                 constraints=[(0.1, 1.0), (0.3, 1.0),
                              (0.5, 1.0), (0.7, 1.0),
                              (0.9, 1.0), (0.0, 1.0)]):
        super(RandomCropImage, self).__init__()
        self.constraints = constraints
        self.scales = scales
        self.max_ratio = max_ratio
        self.max_trial = max_trial

    def __call__(self, sample):
        im = sample['image']
        gt_bbox = sample['gt_bbox']
        gt_class = sample['gt_class']
        im_width = sample['w']
        im_height = sample['h']

        if len(gt_bbox) == 0:
            return sample

        gt_bbox = gt_bbox * ([im_width, im_height] * 2)
        gt_bbox[:, 0] = (gt_bbox[:, 2] + gt_bbox[:, 0]) / 2
        gt_bbox[:, 1] = (gt_bbox[:, 3] + gt_bbox[:, 1]) / 2
        gt_bbox[:, 2] = (gt_bbox[:, 2] - gt_bbox[:, 0]) * 2
        gt_bbox[:, 3] = (gt_bbox[:, 3] - gt_bbox[:, 1]) * 2
        img = Image.fromarray(im)
        crops = [(0, 0, im_width, im_height)]
        for min_iou, max_iou in self.constraints:
            for _ in range(self.max_trial):
                scale = random.uniform(self.scales[0], self.scales[1])
                aspect_ratio = random.uniform(max(1 / self.max_ratio, scale * scale),
                                              min(self.max_ratio, 1 / scale / scale))
                crop_h = int(im_height * scale / np.sqrt(aspect_ratio))
                crop_w = int(im_width * scale * np.sqrt(aspect_ratio))
                crop_x = random.randrange(im_width - crop_w)
                crop_y = random.randrange(im_height - crop_h)
                crop_box = np.array([[(crop_x + crop_w / 2.0) / im_width,
                                    (crop_y + crop_h / 2.0) / im_height,
                                    crop_w / float(im_width), crop_h / float(im_height)]])

                iou = multi_box_iou(crop_box, gt_bbox, xywh=True)
                if min_iou <= iou.min() and max_iou >= iou.max():
                    crops.append((crop_x, crop_y, crop_w, crop_h))
                    break

        while crops:
            crop = crops.pop(np.random.randint(0, len(crops)))
            crop_boxes, crop_labels, box_num = box_crop(gt_bbox, gt_class, crop,
                                                        (im_width, im_height))
            if box_num < 1:
                continue
            img = img.crop((crop[0], crop[1], crop[0] + crop[2],
                            crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
            img = np.asarray(img)
            sample['image'] = img
            sample['gt_bbox'] = crop_boxes
            sample['gt_class'] = crop_labels
            return sample
        return sample


class RandomInterpImage(object):

    def __init__(self, target_size=0, interp=None):
        super(RandomInterpImage, self).__init__()
        self.target_size = target_size
        interp_method = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_CUBIC,
            cv2.INTER_AREA,
            cv2.INTER_LANCZOS4,
        ]
        if not interp or interp not in interp_method:
            self.interp = interp_method[random.randint(0, len(interp_method) - 1)]
        elif interp in interp_method:
            self.interp = interp

    def __call__(self, sample):
        im = sample['image']
        im_width = sample['w']
        im_height = sample['h']

        im_scale_x = self.target_size / float(im_width)
        im_scale_y = self.target_size / float(im_height)
        img = cv2.resize(
            im, None, None,
            fx=im_scale_x,
            fy=im_scale_y,
            interpolation=self.interp
        )
        sample["image"] = img
        return sample


class NormalizeBox(object):

    def __init__(self):
        super(NormalizeBox, self).__init__()

    def __call__(self, sample):
        gt_bbox = sample['gt_bbox']
        width = sample['w']
        height = sample['h']
        for i in range(gt_bbox.shape[0]):
            gt_bbox[i][0] = gt_bbox[i][0] / width
            gt_bbox[i][1] = gt_bbox[i][1] / height
            gt_bbox[i][2] = gt_bbox[i][2] / width
            gt_bbox[i][3] = gt_bbox[i][3] / height
        sample['gt_bbox'] = gt_bbox
        return sample


class Permute(object):

    def __init__(self, to_bgr=False, channel_first=True):
        super(Permute, self).__init__()
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, sample, context=None):
        im = sample["image"]
        if self.channel_first:
            im = im.astype('float32').transpose((2, 0, 1))
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        sample["image"] = im
        return sample


class MixupImage(object):

    def __init__(self, alpha=1.5, beta=1.5):
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            return sample['mixup']
        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        gt_bbox1 = sample['gt_bbox']
        gt_bbox2 = sample['mixup']['gt_bbox']
        gt_bbox = np.concatenate((gt_bbox1, gt_bbox2), axis=0)
        gt_class1 = sample['gt_class']
        gt_class2 = sample['mixup']['gt_class']
        gt_class = np.concatenate((gt_class1, gt_class2), axis=0)
        difficult1 = sample['difficult']
        difficult2 = sample['mixup']['difficult']
        difficult = np.concatenate((difficult1, difficult2), axis=0)

        gt_score1 = sample['gt_score']
        gt_score2 = sample['mixup']['gt_score']
        gt_score = np.concatenate((gt_score1 * factor, gt_score2 * (1. - factor)), axis=0)
        sample['image'] = im
        sample['gt_bbox'] = gt_bbox
        sample['gt_score'] = gt_score
        sample['gt_class'] = gt_class
        sample['difficult'] = difficult
        sample['h'] = im.shape[0]
        sample['w'] = im.shape[1]
        sample['im_size'] = np.array([im.shape[0], im.shape[1]])
        sample.pop('mixup')
        return sample


class BboxXYXY2XYWH(object):

    def __init__(self):
        super(BboxXYXY2XYWH, self).__init__()

    def __call__(self, sample):
        bbox = sample['gt_bbox']
        bbox[:, 0] = (bbox[:, 2] + bbox[:, 0]) / 2
        bbox[:, 1] = (bbox[:, 3] + bbox[:, 1]) / 2
        bbox[:, 2] = (bbox[:, 2] - bbox[:, 0]) * 2
        bbox[:, 3] = (bbox[:, 3] - bbox[:, 1]) * 2
        sample['gt_bbox'] = bbox
        return sample


class BatchRandomReshape(object):

    def __init__(self, shapes=[320, 352, 384, 416, 448, 480, 512, 544, 576, 608]):
        super(BatchRandomReshape, self).__init__()
        self.shapes = shapes

    def __call__(self, batch_data):
        shape = np.random.choice(self.shapes)
        scaled_batch = []
        h, w = batch_data[0][0].shape[1:3]
        scale_x = float(shape) / w
        scale_y = float(shape) / h
        for data in batch_data:
            im = cv2.resize(
                data[0].transpose((1, 2, 0)),
                None,
                None,
                fx=scale_x,
                fy=scale_y,
                interpolation=cv2.INTER_CUBIC)
            scaled_batch.append([im.transpose(2, 0, 1)] + data[1:])
        return scaled_batch
