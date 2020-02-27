import paddle.fluid as fluid

from detection.resnet import ResNet
from detection.yolo_head import YOLOv3Head
from detection.yolov3 import YOLOv3
from process.detect_ops import (eval_results, eval_run, load_pretrained_params,
                                parse_fetches)
from reader.yolo_reader import YoloReader


def eval_yolo(eval_dir, args):
    eval_feed = YoloReader(eval_dir, batch_size=args["batch_size"],
                           test_image_shape=args["image_shape"], mode="eval")
    eval_reader = eval_feed.create_reader()

    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    backbone = ResNet(depth=50, freeze_at=2, norm_type="bn", freeze_norm=False,
                      norm_decay=0.0, variant="d", feature_maps=[3, 4, 5],
                      dcn_v2_stages=[5])
    yolo_head = YOLOv3Head(num_classes=args["num_classes"], anchors=args["anchors"],
                           anchor_masks=args["anchor_masks"], nms_threshold=args["nms_thresh"],
                           nms_keep_topk=args["keep_topk"], score_threshold=args["score_threshold"])
    model = YOLOv3(backbone, yolo_head, freeze_backbone=False)

    startup_prog = fluid.Program()
    main_prog = fluid.Program()
    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            eval_feed_vars = eval_feed.build_inputs(
                image_shape=[None, 3, args["image_shape"], args["image_shape"]]
            )
            eval_fetches = model.eval(eval_feed_vars)

    # load model
    exe.run(startup_prog)
    load_pretrained_params(exe, main_prog, args["weights"], ignore_params=[])

    eval_prog = main_prog.clone(True)
    compile_program = fluid.compiler.CompiledProgram(eval_prog).with_data_parallel()

    extra_keys = []
    eval_keys = list(eval_fetches.keys())
    eval_keys, eval_values, eval_cls = parse_fetches(eval_fetches, eval_prog, extra_keys)
    results = eval_run(eval_reader, exe, compile_program, eval_keys, eval_values)
    map_stats = eval_results(results, args["num_classes"])
    return map_stats


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
    eval_dir = "/data/insects/val"
    eval_yolo(eval_dir, args)
