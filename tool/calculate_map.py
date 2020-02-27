import os
import json
import numpy as np
import xml.etree.ElementTree as ET
from utils.map_utils import DetectionMAP

import argparse


def parse_args():
    parser = argparse.ArgumentParser("Evaluation Parameters")
    parser.add_argument(
        '--anno_dir',
        type=str,
        default='./insects/test/annotations/xmls',
        help='the directory of annotations')
    parser.add_argument(
        '--pred_result',
        type=str,
        default='./pred_results.json',
        help='the file path of prediction results')
    args = parser.parse_args()
    return args


args = parse_args()

# 存放测试集标签的目录
annotations_dir = args.anno_dir

# 存放预测结果的文件
pred_result_file = args.pred_result
# pred_result_file 中保存着测试结果，是包含所有图片预测结果的list，其构成如下
# [[img_name, [[label, score, x1, x2, y1, y2],..., [label, score, x1, x2, y1, y2]]], 
#  [img_name, [[label, score, x1, x2, y1, y2],..., [label, score, x1, x2, y1, y2]]],
#  ...
#  [img_name, [[label, score, x1, x2, y1, y2],..., [label, score, x1, x2, y1, y2]]]]
# list中的每一个元素是一张图片的预测结果，list的总长度等于图片的数目
# 每张图片预测结果的格式是: [img_name, [[label, score, x1, x2, y1, y2],..., [label, score, x1, x2, y1, y2]]]
# 其中第一个元素是图片名称image_name，第二个元素是包含所有预测框的列表
# 预测框列表中每个元素[label, score, x1, x2, y1, y2]描述了一个预测框，
# label是预测框所属类别标签，score是预测框的得分
# x1, x2, y1, y2对应预测框左上角坐标(x1, y1)，右下角坐标(x2, y2)
# 每张图片可能有很多个预测框，则将其全部放在预测框列表中

overlap_thresh = 0.5
map_type = '11point'
insect_names = ['Boerner', 'Leconte', 'Linnaeus', 'acuminatus', 'armandi', 'coleoptera', 'linnaeus']

if __name__ == '__main__':
    cname2cid = {}
    for i, item in enumerate(insect_names):
        cname2cid[item] = i
    filename = pred_result_file
    results = json.load(open(filename))
    num_classes = len(insect_names)
    detection_map = DetectionMAP(class_num=num_classes, overlap_thresh=overlap_thresh,
                                 map_type=map_type, is_bbox_normalized=False,
                                 evaluate_difficult=False)
    for result in results:
        image_name = str(result[0])
        bboxes = np.array(result[1]).astype('float32')
        anno_file = os.path.join(annotations_dir, image_name + '.xml')
        tree = ET.parse(anno_file)
        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), 1), dtype=np.int32)
        difficult = np.zeros((len(objs), 1), dtype=np.int32)
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i][0] = cname2cid[cname]
            _difficult = int(obj.find('difficult').text)
            x1 = float(obj.find('bndbox').find('xmin').text)
            y1 = float(obj.find('bndbox').find('ymin').text)
            x2 = float(obj.find('bndbox').find('xmax').text)
            y2 = float(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            gt_bbox[i] = [x1, y1, x2, y2]
            difficult[i][0] = _difficult
        detection_map.update(bboxes, gt_bbox, gt_class, difficult)

    print("Accumulating evaluatation results...")
    detection_map.accumulate()
    map_stat = 100. * detection_map.get_map()
    print("mAP({:.2f}, {}) = {:.4f}".format(overlap_thresh, map_type, map_stat))
