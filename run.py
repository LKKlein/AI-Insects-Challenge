from detection.infer_yolo import infer_yolo
from classification.infer_cls import infer_cls
from process.post_process import process
import json


if __name__ == "__main__":
    test_dir = "data/insects/test/images"
    detection_args = {
        "anchors": [
            [19, 29], [28, 20], [25, 40],
            [31, 47], [36, 37], [41, 26],
            [47, 66], [48, 33], [67, 53]
        ],
        "anchor_masks": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "keep_topk": 18,
        "nms_thresh": 0.45,
        "score_threshold": 0.1,
        "num_classes": 7,
        "batch_size": 10,
        "image_shape": 608,
        "weights": "models/yolov3_resnet50vd_dcn",
        "use_cuda": True
    }
    cls_args = args = {
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
    total_results = infer_yolo(test_dir, detection_args)
    total_results = infer_cls(cls_args, test_dir, total_results)
    total_results = process(total_results)
    json.dump(total_results, open('pred_results.json', 'w'))
