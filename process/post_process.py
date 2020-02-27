import numpy as np
from utils.box_utils import box_iou_xyxy, get_outer_box, box_area_iou_xyxy


def process(results):
    total_results = []
    for idx, result in enumerate(results):
        image_name = str(result[0])
        bboxes = np.array(result[1]).astype('float32')
        total_boxes = merge_lower_iou(image_name, bboxes,
                                      box_iou_xyxy, iou_thresh=0.45)
        total_boxes = merge_lower_iou(image_name, total_boxes,
                                      box_area_iou_xyxy, iou_thresh=0.9)
        total_boxes = drop_lower_score(image_name, total_boxes, score_thresh=0.45)
        total_results.append([str(image_name), total_boxes])
    return total_results


def merge_lower_iou(image_name, bboxes, iou_method, iou_thresh=0.5):
    total_index = list(range(len(bboxes)))
    total_boxes = []
    while len(total_index) > 0:
        box_i = np.array(bboxes[total_index[0]])
        drop_index = [0]
        for index in range(1, len(total_index)):
            box_j = np.array(bboxes[total_index[index]])
            if box_i[0] != box_j[0]:
                continue
            iou = iou_method(box_i[2:], box_j[2:])
            if iou > iou_thresh:
                box_i[2:] = get_outer_box(box_i[2:], box_j[2:])
                box_i[1] = max(box_i[1], box_j[1])
                drop_index.append(index)

        total_index = [item for idx, item in enumerate(total_index) if idx not in drop_index]
        total_boxes.append(box_i.tolist())
    return total_boxes


def drop_lower_score(image_name, bboxes, score_thresh=0.01):
    total_boxes = []
    for box in bboxes:
        if box[1] > score_thresh:
            total_boxes.append(box)
    return total_boxes
