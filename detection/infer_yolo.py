import json

import numpy as np
import paddle.fluid as fluid

from detection.resnet import ResNet
from detection.yolo_head import YOLOv3Head
from detection.yolov3 import YOLOv3
from process.detect_ops import eval_run, load_pretrained_params, parse_fetches
from reader.yolo_reader import YoloReader


def infer_yolo(test_dir, args):
    test_feed = YoloReader(test_dir, batch_size=args["batch_size"],
                           test_image_shape=args["image_shape"], mode="test")
    test_reader = test_feed.create_reader()

    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    backbone = ResNet(depth=50, freeze_at=2, norm_type="bn", freeze_norm=False,
                      norm_decay=0.0, variant="d", feature_maps=[3, 4, 5],
                      dcn_v2_stages=[5])
    yolov3_head = YOLOv3Head(num_classes=args["num_classes"], anchors=args["anchors"],
                             anchor_masks=args["anchor_masks"],
                             nms_threshold=args["nms_thresh"],
                             nms_keep_topk=args["keep_topk"],
                             score_threshold=args["score_threshold"])
    model = YOLOv3(backbone, yolov3_head, freeze_backbone=False)

    startup_prog = fluid.Program()
    main_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            test_feed_vars = test_feed.build_inputs(
                image_shape=[None, 3, args["image_shape"], args["image_shape"]]
            )
            test_fetches = model.test(test_feed_vars)

    # load model
    exe.run(startup_prog)
    load_pretrained_params(exe, main_prog, args["weights"], ignore_params=[])

    test_prog = main_prog.clone(True)
    compile_program = fluid.compiler.CompiledProgram(test_prog).with_data_parallel()

    extra_keys = []
    test_keys, test_values, test_cls = parse_fetches(test_fetches, test_prog, extra_keys)
    results = eval_run(test_reader, exe, compile_program, test_keys, test_values)
    total_results = []
    map_idx = {0.0: 0.0, 1.0: 6.0, 2.0: 4.0, 3.0: 5.0, 4.0: 2.0, 5.0: 1.0, 6.0: 3.0}
    for res in results:
        bboxes = res["bbox"]
        im_ids = res["im_id"]
        start = 0
        for i in range(len(im_ids)):
            im_name = im_ids[i][0]
            box_len = bboxes[1][0][i]
            cur = np.sum(bboxes[1][0][0:i + 1])
            boxes = bboxes[0][start:cur, :].copy()
            for box in boxes:
                box[0] = map_idx[box[0]]
            start = cur
            assert len(boxes) == box_len, "length wrong!"
            total_results.append([str(im_name), boxes.tolist()])
    total_results = list(sorted(total_results, key=lambda x: x[0]))
    print('processed finished, total {} pictures'.format(len(total_results)))
    return total_results


if __name__ == "__main__":
    args = {
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
    test_dir = "data/insects/test/images"
    total_results = infer_yolo(test_dir, args)
    json.dump(total_results, open('pred_results_infer.json', 'w'))
