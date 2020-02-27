import os
import time

import paddle.fluid as fluid

import sys
sys.path.append("./")

from detection.resnet import ResNet
from detection.yolo_head import YOLOv3Head
from detection.yolov3 import YOLOv3
from reader.yolo_reader import YoloReader
from process.detect_ops import (eval_results, eval_run, load_pretrained_params,
                                parse_fetches, save_params)


def train_yolo(train_dir, args):
    train_feed = YoloReader(train_dir, num_max_boxes=args["num_max_boxes"],
                            shuffle_images=args["shuffle_images"], iters=args["iters"],
                            mixup_epochs=args["mixup_epochs"], batch_size=args["batch_size"],
                            mode="train")
    if args["_eval"]:
        eval_feed = YoloReader(args["eval_dir"], batch_size=args["batch_size"],
                               test_image_shape=args["image_shape"], mode="eval")

    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build program
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            backbone = ResNet(depth=50, freeze_at=4, norm_type="bn",
                              freeze_norm=args["freeze_norm"], norm_decay=0.0, variant="d",
                              feature_maps=[3, 4, 5], dcn_v2_stages=[5])
            yolo_head = YOLOv3Head(num_classes=args["num_classes"], anchors=args["anchors"],
                                   anchor_masks=args["anchor_masks"], norm_decay=0,
                                   freeze_block=args["freeze_block"], freeze_route=args["freeze_route"],
                                   drop_block=False, block_size=3, keep_prob=0.95,
                                   ignore_thresh=args["ignore_thresh"], freeze_norm=args["freeze_norm"],
                                   label_smooth=args["use_label_smooth"])
            model = YOLOv3(backbone, yolo_head, freeze_backbone=args["freeze_backbone"])
            feed_vars = train_feed.build_inputs(image_shape=[None, 3, None, None])
            train_fetches = model.train(feed_vars)
            loss = train_fetches["loss"]

            opt = fluid.optimizer.MomentumOptimizer(
                learning_rate=fluid.layers.cosine_decay(args["lr"],
                    step_each_epoch=args["steps_per_epoch"], epochs=args["iters"] // args["steps_per_epoch"]),
                momentum=args["momentum"],
                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=args["l2_coffe"])
            )
            opt.minimize(loss)

    train_keys, train_values, _ = parse_fetches(train_fetches)

    if args["_eval"]:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                backbone = ResNet(depth=50, freeze_at=4, norm_type="bn",
                                  freeze_norm=False, norm_decay=0.0, variant="d",
                                  feature_maps=[3, 4, 5], dcn_v2_stages=[5])
                yolo_head = YOLOv3Head(num_classes=args["num_classes"],
                                       anchors=args["anchors"],
                                       anchor_masks=args["anchor_masks"],
                                       nms_threshold=args["nms_thresh"],
                                       nms_keep_topk=args["keep_topk"],
                                       score_threshold=args["score_threshold"])
                model = YOLOv3(backbone, yolo_head)
                feed_vars = eval_feed.build_inputs(image_shape=[None, 3, 608, 608])
                eval_fetches = model.eval(feed_vars)
        eval_prog = eval_prog.clone(True)

        eval_reader = eval_feed.create_reader()

        extra_keys = []
        eval_keys, eval_values, eval_cls = parse_fetches(eval_fetches, eval_prog, extra_keys)

        build_strategy = fluid.BuildStrategy()
        exec_strategy = fluid.ExecutionStrategy()
        exec_strategy.num_iteration_per_drop_scope = 1

        exe.run(startup_prog)
        compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
            loss_name=loss.name,
            build_strategy=build_strategy,
            exec_strategy=exec_strategy
        )

        if args["_eval"]:
            compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)

        load_pretrained_params(exe, train_prog, args["pretrain_weights"],
                               ignore_params=args["ignore_weights"])

        train_reader = train_feed.create_reader()
        best_map = [0, 0]

        for idx, data in enumerate(train_reader()):
            loss_result, = exe.run(compiled_train_prog, feed=data, fetch_list=train_values)
            if idx % args["log_iter"] == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print('[{} [TRAIN]] iter {}, output loss: {}'.format(timestring, idx,
                                                                     loss_result[0]))

            if idx % args["save_iter"] == 0:
                if not os.path.exists(args["save_dir"]):
                    os.makedirs(args["save_dir"], exist_ok=True)
                save_name = "yolov3_resnet50vd_dcn_iter_{}".format(idx)

                if args["_eval"]:
                    results = eval_run(eval_reader, exe, compiled_eval_prog, eval_keys, eval_values)
                    map_stat = eval_results(results, args["num_classes"])
                    if map_stat > best_map[1]:
                        best_map = [idx, map_stat]
                        save_path = os.path.join(args["save_dir"],
                                                 "yolov3_resnet50vd_dcn_best_model")
                        save_params(exe, train_prog, save_path)

                    print('[{}] best map: {:.5f} at iter {}'.format(
                        timestring, best_map[1], best_map[0]))

                    if map_stat > 99:
                        save_params(exe, train_prog, os.path.join(args["save_dir"], save_name))


if __name__ == "__main__":
    args = {
        "train_dir": "data/insects/train",
        "eval_dir": "data/insects/val",
        "anchors": [
            [19, 29], [28, 20], [25, 40],
            [31, 47], [36, 37], [41, 26],
            [47, 66], [48, 33], [67, 53]
        ],
        "anchor_masks": [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "num_classes": 7,
        "keep_topk": 100,
        "nms_thresh": 0.45,
        "score_threshold": 0.01,
        "image_shape": 608,
        "ignore_thresh": 0.7,
        "num_max_boxes": 50,
        "lr": 0.0001,
        "l2_coffe": 0.0005,
        "iters": 40000,
        "steps_per_epoch": 170,
        "momentum": 0.9,
        "save_iter": 50,
        "log_iter": 10,
        "batch_size": 12,
        "mixup_epochs": 2500,
        "ignore_weights": ["yolo_output"],
        "pretrain_weights": "pretrain_weights/yolov3_resnet50vd_dcn",
        "save_dir": "models/",
        "freeze_backbone": True,
        "freeze_route": [0],
        "freeze_block": [],
        "freeze_norm": True,
        "map_type": "11point",
        "shuffle_images": True,
        "use_label_smooth": True,
        "use_cuda": True,
        "_eval": True
    }

    train_yolo(args["train_dir"], args)
