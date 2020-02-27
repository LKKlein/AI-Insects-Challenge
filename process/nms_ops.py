import numpy as np
from utils.box_utils import box_iou_xyxy


def nms(bboxes, scores, score_thresh, nms_thresh, pre_nms_topk):
    """
    nms
    """
    inds = np.argsort(scores)
    inds = inds[::-1]
    inds = inds[:pre_nms_topk]
    keep_inds = []
    while(len(inds) > 0):
        cur_ind = inds[0]
        cur_score = scores[cur_ind]
        # if score of the box is less than score_thresh, just drop it
        if cur_score < score_thresh:
            break

        keep = True
        for ind in keep_inds:
            current_box = bboxes[cur_ind]
            remain_box = bboxes[ind]
            iou = box_iou_xyxy(current_box, remain_box)
            if iou > nms_thresh:
                keep = False
                break
        if keep:
            keep_inds.append(cur_ind)
        inds = inds[1:]

    return np.array(keep_inds)


def multiclass_nms(bboxes, scores, score_thresh=0.01, nms_thresh=0.45, pre_nms_topk=1000, pos_nms_topk=100):
    """
    This is for multiclass_nms
    """
    batch_size = bboxes.shape[0]
    class_num = scores.shape[1]
    rets = []
    for i in range(batch_size):
        bboxes_i = bboxes[i]
        scores_i = scores[i]
        ret = []
        for c in range(class_num):
            scores_i_c = scores_i[c]
            keep_inds = nms(bboxes_i, scores_i_c, score_thresh, nms_thresh, pre_nms_topk)
            if len(keep_inds) < 1:
                continue
            keep_bboxes = bboxes_i[keep_inds]
            keep_scores = scores_i_c[keep_inds]
            keep_results = np.zeros([keep_scores.shape[0], 6])
            keep_results[:, 0] = c
            keep_results[:, 1] = keep_scores[:]
            keep_results[:, 2:6] = keep_bboxes[:, :]
            ret.append(keep_results)
        if len(ret) < 1:
            rets.append(ret)
            continue
        ret_i = np.concatenate(ret, axis=0)
        scores_i = ret_i[:, 1]
        if len(scores_i) > pos_nms_topk:
            inds = np.argsort(scores_i)[::-1]
            inds = inds[:pos_nms_topk]
            ret_i = ret_i[inds]

        rets.append(ret_i)

    return rets


def soft_nms_for_cls(dets, sigma, thres, normalized=True):
    """soft_nms_for_cls"""
    dets_final = []
    while len(dets) > 0:
        maxpos = np.argmax(dets[:, 0])
        dets_final.append(dets[maxpos].copy())
        ts, tx1, ty1, tx2, ty2 = dets[maxpos]
        scores = dets[:, 0]
        x1 = dets[:, 1]
        y1 = dets[:, 2]
        x2 = dets[:, 3]
        y2 = dets[:, 4]
        eta = 0 if normalized else 1
        areas = (x2 - x1 + eta) * (y2 - y1 + eta)
        xx1 = np.maximum(tx1, x1)
        yy1 = np.maximum(ty1, y1)
        xx2 = np.minimum(tx2, x2)
        yy2 = np.minimum(ty2, y2)
        w = np.maximum(0.0, xx2 - xx1 + eta)
        h = np.maximum(0.0, yy2 - yy1 + eta)
        inter = w * h
        ovr = inter / (areas + areas[maxpos] - inter)
        weight = np.exp(-(ovr * ovr) / sigma)
        scores = scores * weight
        idx_keep = np.where(scores >= thres)
        dets[:, 0] = scores
        dets = dets[idx_keep]
    dets_final = np.array(dets_final).reshape(-1, 5)
    return dets_final


def multiclass_softnms(bboxes, scores,
                       softnms_thres=0.01,
                       keep_top_k=1000,
                       softnms_sigma=0.05,
                       normlized=False,
                       background_label=-1):
    batch_size = bboxes.shape[0]
    class_nums = scores.shape[1]

    start_idx = 1 if background_label == 0 else 0
    res = []
    for i in range(batch_size):
        bboxes_i = bboxes[i]
        scores_i = scores[i]
        cls_boxes = [[] for _ in range(class_nums)]
        cls_ids = [[] for _ in range(class_nums)]
        for j in range(start_idx, class_nums):
            inds = np.where(scores_i[j, :] >= softnms_thres)[0]
            scores_j = scores_i[j, inds]
            rois_j = bboxes_i[inds]
            dets_j = np.hstack((scores_j[:, np.newaxis], rois_j)).astype(np.float32, copy=False)
            cls_rank = np.argsort(-dets_j[:, 0])
            dets_j = dets_j[cls_rank]

            cls_boxes[j] = soft_nms_for_cls(dets_j, sigma=softnms_sigma,
                                            thres=softnms_thres, normalized=normlized)
            cls_ids[j] = np.array([j] * cls_boxes[j].shape[0]).reshape(-1, 1)

        cls_boxes = np.vstack(cls_boxes[start_idx:])
        cls_ids = np.vstack(cls_ids[start_idx:])
        pred_result = np.hstack([cls_ids, cls_boxes])

        # Limit to max_per_image detections **over all classes**
        image_scores = cls_boxes[:, 0]
        if len(image_scores) > keep_top_k:
            image_thresh = np.sort(image_scores)[-keep_top_k]
            keep = np.where(cls_boxes[:, 0] >= image_thresh)[0]
            pred_result = pred_result[keep, :]
        res.append(pred_result)
    return res
