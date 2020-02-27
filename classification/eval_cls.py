import time

import numpy as np
import paddle.fluid as fluid

from classification.se_resnet_vd import SE_ResNet50_vd
from reader.cls_reader import DataReader


def eval_cls(args):
    image = fluid.data(name='image', shape=[None] + args["image_shape"], dtype='float32')
    label = fluid.data(name='label', shape=[None, 1], dtype='int64')

    model = SE_ResNet50_vd()
    out = model.net(input=image, class_dim=args["num_classes"])

    cost, pred = fluid.layers.softmax_with_cross_entropy(out, label, return_softmax=True)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)
    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name, pred.name]
    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    exe.run(fluid.default_startup_program())
    places = fluid.framework.cuda_places()
    compiled_program = fluid.compiler.CompiledProgram(
        test_program).with_data_parallel(places=places)
    fluid.io.load_persistables(exe, args["weights"])

    data_reader = DataReader(args["batch_size"])
    val_reader = data_reader.val(settings=args)

    feeder = fluid.DataFeeder(place=places, feed_list=[image, label])

    test_info = [[], [], []]
    cnt = 0
    parallel_data = []
    place_num = fluid.core.get_cuda_device_count()
    real_iter = 0

    for batch_id, data in enumerate(val_reader()):
        # image data and label
        image_data = [items[0:2] for items in data]
        parallel_data.append(image_data)
        if place_num == len(parallel_data):
            t1 = time.time()
            loss_set, acc1_set, acc5_set, pred_set = exe.run(
                compiled_program,
                fetch_list=fetch_list,
                feed=list(feeder.feed_parallel(parallel_data, place_num)))
            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss_set)
            acc1 = np.mean(acc1_set)
            acc5 = np.mean(acc5_set)
            test_info[0].append(loss * len(data))
            test_info[1].append(acc1 * len(data))
            test_info[2].append(acc5 * len(data))
            cnt += len(data)
            if batch_id % args["log_iter"] == 0:
                info = "batch {},loss {:.5f}, acc1 {:.5f}, acc5 {:.5f}, time {:.2f} sec".format(
                    real_iter, loss, acc1, acc5, period)
                print(info)

            parallel_data = []
            real_iter += 1

    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt

    info = "Test_loss {:.5f}, test_acc1 {:.5f}, test_acc5 {:.5f}".format(
        test_loss, test_acc1, test_acc5)
    print(info)


if __name__ == "__main__":
    args = {
        "data_dir": "data/insect_cls",
        "batch_size": 64,
        "num_classes": 7,
        "log_iter": 20,
        "weights": "/world/data-c40/lvkun/self/projects/cls_insect/se_resnet50_vd/pretrain2/SE_ResNet50_vd/0",
        "interpolation": None,
        "resize_short_size": 128,
        "image_mean": [0.8937, 0.9031, 0.8988],
        "image_std": [0.19, 0.1995, 0.2022],
        "image_shape": [3, 112, 112],
        "lower_scale": 0.08,
        "lower_ratio": 0.75,
        "upper_ratio": 1.33,
        "use_cuda": True
    }

    eval_cls(args)
