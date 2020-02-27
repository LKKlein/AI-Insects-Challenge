import os
import shutil
import time

import numpy as np
import paddle.fluid as fluid

from utils.map_utils import DetectionMAP


def load_pretrained_params(exe, program, path, ignore_params=["yolo_output"]):

    def if_exist(var):
        var_path = os.path.join(path, var.name)
        exist = os.path.exists(var_path)
        can_load = exist and all([False if param_name in var_path else True
                                  for param_name in ignore_params])
        if not can_load and any([param_name in var_path for param_name in ignore_params]):
            print("ignore params: ", var.name)
        return can_load

    fluid.io.load_vars(exe, path, predicate=if_exist, main_program=program)


def save_params(exe, prog, path):
    """
    Load model from the given path.
    Args:
        exe (fluid.Executor): The fluid.Executor object.
        prog (fluid.Program): save weight from which Program object.
        path (string): the path to save model.
    """
    if os.path.isdir(path):
        shutil.rmtree(path)
    print('Save model to {}.'.format(path))
    fluid.io.save_persistables(exe, path, prog)


def parse_fetches(fetches, prog=None, extra_keys=None):
    """
    Parse fetch variable infos from model fetches,
    values for fetch_list and keys for stat
    """
    keys, values = [], []
    cls = []
    for k, v in fetches.items():
        if hasattr(v, 'name'):
            keys.append(k)
            v.persistable = True
            values.append(v.name)
        else:
            cls.append(v)

    if prog is not None and extra_keys is not None:
        for k in extra_keys:
            try:
                v = fluid.framework._get_var(k, prog)
                keys.append(k)
                values.append(v.name)
            except Exception:
                print("{} not exist in framework.".format(k))
                pass

    return keys, values, cls


def eval_run(reader, exe, prog, keys, values):
    results = []
    iter_id = 0
    images_num = 0
    start_time = time.time()
    for idx, data in enumerate(reader()):
        outs = exe.run(prog, feed=data, fetch_list=values, return_numpy=False)
        res = {
            k: (np.array(v), v.recursive_sequence_lengths())
            for k, v in zip(keys, outs)
        }
        res.update(data)
        results.append(res)
        if iter_id % 100 == 0:
            print('Test iter {}'.format(iter_id))
        iter_id += 1
        images_num += len(res['bbox'][1][0])

    print('Test finish iter {}'.format(iter_id))
    end_time = time.time()
    fps = images_num / (end_time - start_time)
    print('Total number of images: {}, inference time: {} fps.'.format(images_num, fps))
    return results


def eval_results(results, num_classes, overlap_thresh=0.5, map_type="11point"):
    detection_map = DetectionMAP(class_num=num_classes, overlap_thresh=overlap_thresh,
                                 map_type=map_type, is_bbox_normalized=False,
                                 evaluate_difficult=False)
    images = 0
    for res in results:
        bboxes = res["bbox"]
        im_ids = res["im_id"]
        gt_bboxs = res["gt_bbox"]
        gt_classes = res["gt_class"]
        difficults = res["difficult"]
        objects = res["objects"]
        start = 0
        images += len(im_ids)
        for i in range(len(im_ids)):
            box_len = bboxes[1][0][i]
            cur = np.sum(bboxes[1][0][0:i + 1])
            boxes = bboxes[0][start:cur, :].copy()
            start = cur
            assert len(boxes) == box_len, "length wrong!"
            obj_len = objects[i][0]
            gt_bbox = gt_bboxs[i][:obj_len, :]
            gt_class = gt_classes[i][:obj_len].reshape(-1, 1)
            difficult = difficults[i][:obj_len].reshape(-1, 1)
            detection_map.update(boxes, gt_bbox, gt_class, difficult)

    print("eval finish! Total images: {}".format(images))
    print("Accumulating evaluatation results...")
    detection_map.accumulate()
    map_stat = 100. * detection_map.get_map()
    print("mAP({:.2f}, {}) = {:.5f}".format(overlap_thresh, map_type, map_stat))
    return map_stat
