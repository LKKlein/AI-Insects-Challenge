import os
import numpy as np
from tqdm import tqdm
import cv2
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
    for fname in filenames:
        fid = fname.split('.')[0]
        fpath = os.path.join(datadir, 'annotations', 'xmls', fname)
        img_file = os.path.join(datadir, 'images', fid + '.jpeg')
        tree = ET.parse(fpath)

        objs = tree.findall('object')
        im_w = int(tree.find('size').find('width').text)
        im_h = int(tree.find('size').find('height').text)
        box = []
        label = []
        
        for i, obj in enumerate(objs):
            cname = obj.find('name').text
            x1 = int(obj.find('bndbox').find('xmin').text)
            y1 = int(obj.find('bndbox').find('ymin').text)
            x2 = int(obj.find('bndbox').find('xmax').text)
            y2 = int(obj.find('bndbox').find('ymax').text)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(im_w - 1, x2)
            y2 = min(im_h - 1, y2)
            box.append([x1, y1, x2, y2])
            label.append(cname2cid[cname])

        voc_rec = {
            'im_file': img_file,
            'im_id': fid,
            'gt_class': label,
            'gt_bbox': box,
        }
        records.append(voc_rec)
    return records


def generate_data(save_dir, records, mode="train"):
    im_out = []

    for record in tqdm(records):
        img = cv2.imread(record["im_file"])
        fid = record["im_id"]
        for i, l in enumerate(record["gt_class"]):
            box = record["gt_bbox"][i]
            im = img[box[1]: box[3], box[0]: box[2]]
            fname = '{}/{}/{}_{}.jpeg'.format(save_dir, mode, str(fid), str(i))
            cv2.imwrite(fname, im)
            im_out.append("{} {}".format(fname, l))

    with open("{}/val_list.txt".format(save_dir), "w") as f:
        f.write("\n".join(im_out))


if __name__ == "__main__":
    cname2cid = get_insect_names()
    save_dir = "data/insect_cls"
    TRAINDIR = "data/insects/train"
    train_records = get_annotations(cname2cid, TRAINDIR)
    generate_data(save_dir, train_records, "train")

    VALIDDIR = "data/insects/val"
    val_records = get_annotations(cname2cid, VALIDDIR)
    generate_data(save_dir, val_records, "val")
