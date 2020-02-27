import json
import math
import os

import cv2
import numpy as np
import paddle.fluid as fluid

from classification.se_resnet_vd import SE_ResNet50_vd
from reader.cls_reader import DataReader


def build_model(args):
    image = fluid.data(name='image', shape=[None] + args["image_shape"], dtype='float32')

    model = SE_ResNet50_vd()
    out = model.net(input=image, class_dim=args["num_classes"])
    out = fluid.layers.softmax(out)

    test_program = fluid.default_main_program().clone(for_test=True)
    fetch_list = [out.name]
    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    fluid.io.load_persistables(exe, args["weights"])

    feeder = fluid.DataFeeder([image], place)
    return exe, test_program, fetch_list, feeder


def predict_cls(args, exe, test_program, fetch_list, feeder):
    data_reader = DataReader(args["batch_size"])
    test_reader = data_reader.test(settings=args)

    outs = []
    for batch_id, data in enumerate(test_reader()):
        image_data = [items[0:1] for items in data]
        result = exe.run(test_program, fetch_list=fetch_list,
                         feed=feeder.feed(image_data))
        for i, res in enumerate(result[0]):
            pred_label = np.argsort(res)[::-1][:1]
            outs.append([int(pred_label), float(res[pred_label])])
    return outs


def infer_cls(args, image_root, total_datas):
    print("start to classify box object...")
    exe, test_program, fetch_list, feeder = build_model(args)

    total_results = []
    map_idx = {0: 0.0, 1: 6.0, 2: 4.0, 3: 5.0, 4: 2.0, 5: 1.0, 6: 3.0}
    for idx, result in enumerate(total_datas):
        image_name = str(result[0])
        bboxes = np.array(result[1]).astype('float32')
        img = cv2.imread(os.path.join(image_root, "{}.jpeg".format(image_name)))
        images = []
        for bbox in bboxes:
            x1, y1 = int(bbox[2]) - 1, int(bbox[3]) - 1
            x2, y2 = int(math.ceil(bbox[4])) + 1, int(math.ceil(bbox[5])) + 1
            images.append(
                img[y1: y2, x1: x2]
            )
        args["images"] = images
        args["images_num"] = len(images)
        results = predict_cls(args, exe, test_program, fetch_list, feeder)
        total_bbox = []
        for result, box in list(zip(results, bboxes.tolist())):
            result = [map_idx[result[0]], result[1]]
            out_box = result + box[2:]
            total_bbox.append(out_box)
        total_bbox = list(sorted(total_bbox, key=lambda x: x[0]))
        total_results.append([str(image_name), total_bbox])
    print("classification finished! Total number of images: {}".format(len(total_results)))
    return total_results


if __name__ == "__main__":
    args = {
        "images": [],
        "images_num": 0,
        "num_classes": 7,
        "batch_size": 64,
        "weights": "models/se_resnet50_vd",
        "ignore_weights": [],
        "interpolation": None,
        "resize_short_size": 128,
        "image_mean": [0.8937, 0.9031, 0.8988],
        "image_std": [0.1900, 0.1995, 0.2022],
        "image_shape": [3, 112, 112],
        "lower_scale": 0.08,
        "lower_ratio": 0.75,
        "upper_ratio": 1.33,
        "use_cuda": True
    }

    im_root = "data/insects/test/images/"
    data = json.load(open("pred_results_infer.json"))
    total_results = infer_cls(args, im_root, data)
    json.dump(total_results, open('pred_results_adjust.json', 'w'))
