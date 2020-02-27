# -*- coding: utf-8 -*-

# 此文件中包含了一些box相关的计算函数

import numpy as np


def get_outer_box(box1, box2):
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    if x1min < x2min:
        x1 = x1min
    else:
        x1 = x2min

    if y1min < y2min:
        y1 = y1min
    else:
        y1 = y2min

    if x1max < x2max:
        x2 = x2max
    else:
        x2 = x1max

    if y1max < y2max:
        y2 = y2max
    else:
        y2 = y1max
    return np.array([x1, y1, x2, y2])


def box_ciou_xyxy(boxes1, boxes2):
    # cal the box's area of boxes1 and boxess
    boxes1Area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2Area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # cal Intersection
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1Area + boxes2Area - inter_area
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    # cal outer boxes
    outer_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    outer = np.maximum(outer_right_down - outer_left_up, 0.0)
    outer_diagonal_line = np.square(outer[..., 0]) + np.square(outer[..., 1])

    # cal center distance
    boxes1_center = (boxes1[..., :2] + boxes1[..., 2:]) * 0.5
    boxes2_center = (boxes2[..., :2] + boxes2[..., 2:]) * 0.5
    center_dis = np.square(boxes1_center[..., 0] - boxes2_center[..., 0]) +\
        np.square(boxes1_center[..., 1] - boxes2_center[..., 1])

    # cal penalty term
    # cal width,height
    boxes1_size = np.maximum(boxes1[..., 2:] - boxes1[..., :2], 0.0)
    boxes2_size = np.maximum(boxes2[..., 2:] - boxes2[..., :2], 0.0)
    v = (4.0 / np.square(np.pi)) * np.square((
        np.arctan((boxes1_size[..., 0] / boxes1_size[..., 1])) -
        np.arctan((boxes2_size[..., 0] / boxes2_size[..., 1]))))
    alpha = v / (1 - ious + v)

    # cal ciou
    cious = ious - (center_dis / outer_diagonal_line + alpha * v)

    return cious


def box_diou_xyxy(box1, box2):
    eps = 1.e-10

    x1, y1, x2, y2 = box1[0], box1[1], box1[2], box1[3]
    x1g, y1g, x2g, y2g = box2[0], box2[1], box2[2], box2[3]

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    cxg = (x1g + x2g) / 2
    cyg = (y1g + y2g) / 2

    x2 = np.maximum(x1, x2)
    y2 = np.maximum(y1, y2)

    # A or B
    xc1 = np.minimum(x1, x1g)
    yc1 = np.minimum(y1, y1g)
    xc2 = np.maximum(x2, x2g)
    yc2 = np.maximum(y2, y2g)

    # DIOU term
    dist_intersection = (cx - cxg)**2 + (cy - cyg)**2
    dist_union = (xc2 - xc1)**2 + (yc2 - yc1)**2
    diou_term = (dist_intersection + eps) / (dist_union + eps)
    return diou_term

def box_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    union = s1 + s2 - intersection
    # 计算交并比
    iou = intersection / union
    return iou


def box_area_iou_xyxy(box1, box2):
    # 获取box1左上角和右下角的坐标
    x1min, y1min, x1max, y1max = box1[0], box1[1], box1[2], box1[3]
    # 计算box1的面积
    s1 = (y1max - y1min + 1.) * (x1max - x1min + 1.)
    # 获取box2左上角和右下角的坐标
    x2min, y2min, x2max, y2max = box2[0], box2[1], box2[2], box2[3]
    # 计算box2的面积
    s2 = (y2max - y2min + 1.) * (x2max - x2min + 1.)

    # 计算相交矩形框的坐标
    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    # 计算相交矩形行的高度、宽度、面积
    inter_h = np.maximum(ymax - ymin + 1., 0.)
    inter_w = np.maximum(xmax - xmin + 1., 0.)
    intersection = inter_h * inter_w
    # 计算相并面积
    # union = s1 + s2 - intersection
    # 计算交并比
    # iou = intersection / union
    return max(intersection / s1, intersection / s2)


# 计算IoU，矩形框的坐标形式为xywh
def box_iou_xywh(box1, box2):
    x1min, y1min = box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0
    x1max, y1max = box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0
    s1 = box1[2] * box1[3]

    x2min, y2min = box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0
    x2max, y2max = box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0
    s2 = box2[2] * box2[3]

    xmin = np.maximum(x1min, x2min)
    ymin = np.maximum(y1min, y2min)
    xmax = np.minimum(x1max, x2max)
    ymax = np.minimum(y1max, y2max)
    inter_h = np.maximum(ymax - ymin, 0.)
    inter_w = np.maximum(xmax - xmin, 0.)
    intersection = inter_h * inter_w

    union = s1 + s2 - intersection
    iou = intersection / union
    return iou


def multi_box_iou(box1, box2, xywh=True):
    """
    In this case, box1 or box2 can contain multi boxes.
    Only two cases can be processed in this method:
       1, box1 and box2 have the same shape, box1.shape == box2.shape
       2, either box1 or box2 contains only one box, len(box1) == 1 or len(box2) == 1
    If the shape of box1 and box2 does not match, and both of them contain multi boxes, it will be wrong.
    """
    assert box1.shape[-1] == 4, "Box1 shape[-1] should be 4."
    assert box2.shape[-1] == 4, "Box2 shape[-1] should be 4."

    b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
    b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
    b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1 = np.maximum(b1_x1, b2_x1)
    inter_x2 = np.minimum(b1_x2, b2_x2)
    inter_y1 = np.maximum(b1_y1, b2_y1)
    inter_y2 = np.minimum(b1_y2, b2_y2)
    inter_w = inter_x2 - inter_x1
    inter_h = inter_y2 - inter_y1
    inter_w = np.clip(inter_w, a_min=0., a_max=None)
    inter_h = np.clip(inter_h, a_min=0., a_max=None)

    inter_area = inter_w * inter_h
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area)


def box_crop(boxes, labels, crop, img_shape):
    x, y, w, h = map(float, crop)
    im_w, im_h = map(float, img_shape)

    boxes = boxes.copy()
    boxes[:, 0], boxes[:, 2] = (boxes[:, 0] - boxes[:, 2] / 2) * im_w, (
        boxes[:, 0] + boxes[:, 2] / 2) * im_w
    boxes[:, 1], boxes[:, 3] = (boxes[:, 1] - boxes[:, 3] / 2) * im_h, (
        boxes[:, 1] + boxes[:, 3] / 2) * im_h

    crop_box = np.array([x, y, x + w, y + h])
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    mask = np.logical_and(crop_box[:2] <= centers, centers <= crop_box[2:]).all(
        axis=1)

    boxes[:, :2] = np.maximum(boxes[:, :2], crop_box[:2])
    boxes[:, 2:] = np.minimum(boxes[:, 2:], crop_box[2:])
    boxes[:, :2] -= crop_box[:2]
    boxes[:, 2:] -= crop_box[:2]

    mask = np.logical_and(mask, (boxes[:, :2] < boxes[:, 2:]).all(axis=1))
    boxes = boxes * np.expand_dims(mask.astype('float32'), axis=1)
    labels = labels * mask.astype('float32')
    boxes[:, 0], boxes[:, 2] = boxes[:, 0] / w, boxes[:, 2] / w
    boxes[:, 1], boxes[:, 3] = boxes[:, 1] / h, boxes[:, 3] / h

    return boxes, labels, mask.sum()


def meet_emit_constraint(src_bbox, sample_bbox):
    center_x = (src_bbox[2] + src_bbox[0]) / 2
    center_y = (src_bbox[3] + src_bbox[1]) / 2
    if center_x >= sample_bbox[0] and \
            center_x <= sample_bbox[2] and \
            center_y >= sample_bbox[1] and \
            center_y <= sample_bbox[3]:
        return True
    return False


def is_overlap(object_bbox, sample_bbox):
    if object_bbox[0] >= sample_bbox[2] or \
       object_bbox[2] <= sample_bbox[0] or \
       object_bbox[1] >= sample_bbox[3] or \
       object_bbox[3] <= sample_bbox[1]:
        return False
    else:
        return True

def clip_bbox(src_bbox):
    src_bbox[0] = max(min(src_bbox[0], 1.0), 0.0)
    src_bbox[1] = max(min(src_bbox[1], 1.0), 0.0)
    src_bbox[2] = max(min(src_bbox[2], 1.0), 0.0)
    src_bbox[3] = max(min(src_bbox[3], 1.0), 0.0)
    return src_bbox


def bbox_area(src_bbox):
    if src_bbox[2] < src_bbox[0] or src_bbox[3] < src_bbox[1]:
        return 0.
    else:
        width = src_bbox[2] - src_bbox[0]
        height = src_bbox[3] - src_bbox[1]
        return width * height


def filter_and_process(sample_bbox, bboxes, labels, scores=None):
    new_bboxes = []
    new_labels = []
    new_scores = []
    for i in range(len(bboxes)):
        new_bbox = [0, 0, 0, 0]
        obj_bbox = [bboxes[i][0], bboxes[i][1], bboxes[i][2], bboxes[i][3]]
        if not meet_emit_constraint(obj_bbox, sample_bbox):
            continue
        if not is_overlap(obj_bbox, sample_bbox):
            continue
        sample_width = sample_bbox[2] - sample_bbox[0]
        sample_height = sample_bbox[3] - sample_bbox[1]
        new_bbox[0] = (obj_bbox[0] - sample_bbox[0]) / sample_width
        new_bbox[1] = (obj_bbox[1] - sample_bbox[1]) / sample_height
        new_bbox[2] = (obj_bbox[2] - sample_bbox[0]) / sample_width
        new_bbox[3] = (obj_bbox[3] - sample_bbox[1]) / sample_height
        new_bbox = clip_bbox(new_bbox)
        if bbox_area(new_bbox) > 0:
            new_bboxes.append(new_bbox)
            new_labels.append([labels[i][0]])
            if scores is not None:
                new_scores.append([scores[i][0]])
    bboxes = np.array(new_bboxes)
    labels = np.array(new_labels)
    scores = np.array(new_scores)
    return bboxes, labels, scores


def generate_sample_bbox(sampler):
    scale = np.random.uniform(sampler[2], sampler[3])
    aspect_ratio = np.random.uniform(sampler[4], sampler[5])
    aspect_ratio = max(aspect_ratio, (scale**2.0))
    aspect_ratio = min(aspect_ratio, 1 / (scale**2.0))
    bbox_width = scale * (aspect_ratio**0.5)
    bbox_height = scale / (aspect_ratio**0.5)
    xmin_bound = 1 - bbox_width
    ymin_bound = 1 - bbox_height
    xmin = np.random.uniform(0, xmin_bound)
    ymin = np.random.uniform(0, ymin_bound)
    xmax = xmin + bbox_width
    ymax = ymin + bbox_height
    sampled_bbox = [xmin, ymin, xmax, ymax]
    return sampled_bbox


def jaccard_overlap(sample_bbox, object_bbox):
    if sample_bbox[0] >= object_bbox[2] or \
        sample_bbox[2] <= object_bbox[0] or \
        sample_bbox[1] >= object_bbox[3] or \
        sample_bbox[3] <= object_bbox[1]:
        return 0
    intersect_xmin = max(sample_bbox[0], object_bbox[0])
    intersect_ymin = max(sample_bbox[1], object_bbox[1])
    intersect_xmax = min(sample_bbox[2], object_bbox[2])
    intersect_ymax = min(sample_bbox[3], object_bbox[3])
    intersect_size = (intersect_xmax - intersect_xmin) * (
        intersect_ymax - intersect_ymin)
    sample_bbox_size = bbox_area(sample_bbox)
    object_bbox_size = bbox_area(object_bbox)
    overlap = intersect_size / (
        sample_bbox_size + object_bbox_size - intersect_size)
    return overlap


def satisfy_sample_constraint(sampler,
                              sample_bbox,
                              gt_bboxes,
                              satisfy_all=False):
    if sampler[6] == 0 and sampler[7] == 0:
        return True
    satisfied = []
    for i in range(len(gt_bboxes)):
        object_bbox = [
            gt_bboxes[i][0], gt_bboxes[i][1], gt_bboxes[i][2], gt_bboxes[i][3]
        ]
        overlap = jaccard_overlap(sample_bbox, object_bbox)
        if sampler[6] != 0 and \
                overlap < sampler[6]:
            satisfied.append(False)
            continue
        if sampler[7] != 0 and \
                overlap > sampler[7]:
            satisfied.append(False)
            continue
        satisfied.append(True)
        if not satisfy_all:
            return True

    if satisfy_all:
        return np.all(satisfied)
    else:
        return False