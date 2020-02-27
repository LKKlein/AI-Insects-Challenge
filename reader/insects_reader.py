# -*- coding: utf-8 -*-

# 此文件中包含昆虫数据集读取相关的函数

import os
import numpy as np
import xml.etree.ElementTree as ET

# 昆虫名称列表
INSECT_NAMES = ['Boerner', 'linnaeus', 'armandi', 'coleoptera',
                'Linnaeus', 'Leconte', 'acuminatus']


def get_insect_names():
    """ 昆虫名字到数字类别的映射关系
    return a dict, as following,
        {'Boerner': 0,
         'linnaeus': 1,
         'armandi': 2,
         'coleoptera': 3,
         'Linnaeus': 4,
         'Leconte': 5,
         'acuminatus': 6,
        }
    {0: 0, 1: 6, 2: 4, 3: 5, 4: 2, 5: 1, 6: 3}
    It can map the insect name into an integer label.
    """
    insect_category2id = {}
    for i, item in enumerate(INSECT_NAMES):
        insect_category2id[item] = i

    return insect_category2id


def get_annotations(cname2cid, datadir):
    """获取昆虫标注信息"""
    filenames = os.listdir(os.path.join(datadir, 'annotations', 'xmls'))
    records = []
    ct = 0
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')
        tree = ET.parse(fpath)

        im_id = np.array([fid])

        objs = tree.findall('object')
        im_w = float(tree.find('size').find('width').text)
        im_h = float(tree.find('size').find('height').text)
        gt_bbox = np.zeros((len(objs), 4), dtype=np.float32)
        gt_class = np.zeros((len(objs), ), dtype=np.int32)
        gt_score = np.ones((len(objs), ), dtype=np.float32)
        is_crowd = np.zeros((len(objs), ), dtype=np.int32)
        difficult = np.zeros((len(objs), ), dtype=np.int32)
        objects = np.array([len(objs)])
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            gt_class[i] = cname2cid[cname]
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
            is_crowd[i] = 0
            difficult[i] = _difficult

        voc_rec = {
            'im_file': img_file,
            'im_id': im_id,
            'objects': objects,
            'im_size': np.array([im_h, im_w]),
            'h': im_h,
            'w': im_w,
            'is_crowd': is_crowd,
            'gt_class': gt_class,
            'gt_score': gt_score,
            'gt_bbox': gt_bbox,
            'gt_poly': [],
            'difficult': difficult
        }
        if len(objs) != 0:
            records.append(voc_rec)
        ct += 1
    return records


def get_test_images(datadir):
    filenames = os.listdir(datadir)
    records = []
    for fname in filenames:
        fid = fname.split('.')[0]
        img_file = os.path.join(datadir, fid + '.jpeg')
        im_id = np.array([fid])
        rec = {
            'im_file': img_file,
            'im_id': im_id
        }
        records.append(rec)
    return records
