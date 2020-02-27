import os
import time

import numpy as np
import paddle.fluid as fluid

from classification.se_resnet_vd import SE_ResNet50_vd
from process.detect_ops import load_pretrained_params, save_params
from reader.cls_reader import DataReader


def train_cls(args):
    data_reader = DataReader(args["batch_size"])

    use_cuda = args["use_cuda"] or fluid.core.is_compiled_with_cuda()
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)

    # build program
    startup_prog = fluid.Program()
    train_prog = fluid.Program()
    with fluid.program_guard(train_prog, startup_prog):
        with fluid.unique_name.guard():
            train_image = fluid.data(name='image', shape=[None] + args["image_shape"],
                                     dtype='float32')
            train_label = fluid.data(name='label', shape=[None, 1], dtype='int64')
            model = SE_ResNet50_vd()
            net_out = model.net(train_image, class_dim=args["num_classes"])
            train_loss, train_pred = fluid.layers.softmax_with_cross_entropy(net_out, train_label,
                                                                             return_softmax=True)
            avg_train_loss = fluid.layers.mean(x=train_loss)
            train_acc = fluid.layers.accuracy(input=train_pred, label=train_label, k=1)

            opt = fluid.optimizer.AdamOptimizer(
                learning_rate=fluid.layers.cosine_decay(args["lr"], step_each_epoch=200, epochs=300),
                regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=args["l2_decay"])
            )
            opt.minimize(avg_train_loss)

            train_feeder = fluid.DataFeeder(place=place, feed_list=[train_image, train_label])

    train_fetches = [avg_train_loss.name, train_acc.name]

    if args["_eval"]:
        eval_prog = fluid.Program()
        with fluid.program_guard(eval_prog, startup_prog):
            with fluid.unique_name.guard():
                eval_image = fluid.data(name='image', shape=[None] + args["image_shape"],
                                        dtype='float32')
                eval_label = fluid.data(name='label', shape=[None, 1], dtype='int64')
                model = SE_ResNet50_vd()
                eval_out = model.net(eval_image, class_dim=args["num_classes"])
                eval_loss, eval_pred = fluid.layers.softmax_with_cross_entropy(eval_out, eval_label,
                                                                               return_softmax=True)
                avg_eval_loss = fluid.layers.mean(x=eval_loss)
                eval_acc = fluid.layers.accuracy(input=eval_pred, label=eval_label, k=1)
                eval_feeder = fluid.DataFeeder(place=place, feed_list=[train_image, eval_label])
        eval_prog = eval_prog.clone(True)

        eval_reader = data_reader.val(settings=args)
        eval_fetches = [avg_eval_loss.name, eval_acc.name]

    build_strategy = fluid.BuildStrategy()
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1

    exe.run(startup_prog)
    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name=train_loss.name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy
    )

    if args["_eval"]:
        compiled_eval_prog = fluid.compiler.CompiledProgram(eval_prog)

    load_pretrained_params(exe, train_prog, args["pretrain_weights"],
                           ignore_params=args["ignore_weights"])

    train_reader = data_reader.train(settings=args)
    best_result = [0, 1000, 0]

    for epoch in range(args["num_epochs"]):
        for idx, data in enumerate(train_reader()):
            image_data = [items[0:2] for items in data]
            loss_result, acc_result = exe.run(compiled_train_prog,
                                              feed=train_feeder.feed(image_data),
                                              fetch_list=train_fetches)
            if idx % args["log_iter"] == 0:
                timestring = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
                print('[{}] ==> Train <== iter {}, loss: {:.4f}, acc: {:.4f}'.format(
                    timestring, idx, np.mean(loss_result), np.mean(acc_result)))

        if epoch % args["save_step"] == 0:
            if not os.path.exists(args["model_save_dir"]):
                os.makedirs(args["model_save_dir"], exist_ok=True)
            save_name = "se_resnet50_vd_{}".format(idx)

            if args["_eval"]:
                data_len = 0
                eval_losses = []
                eval_accs = []
                for jdx, eval_data in enumerate(eval_reader()):
                    eval_image_data = [items[0:2] for items in eval_data]
                    out_loss, out_acc = exe.run(compiled_eval_prog,
                                                feed=eval_feeder.feed(eval_image_data),
                                                fetch_list=eval_fetches)
                    data_len += len(eval_data)
                    eval_accs.append(np.mean(out_acc) * len(eval_data))
                    eval_losses.append(np.mean(out_loss) * len(eval_data))

                final_acc = np.sum(eval_accs) / data_len
                final_loss = np.sum(eval_losses) / data_len
                if final_acc > best_result[2]:
                    best_result = [epoch, final_loss, final_acc]
                    save_path = os.path.join(args["model_save_dir"], "se_resnet50_vd_best_model")
                    save_params(exe, train_prog, save_path)

                print('[{}] ++++++ best acc: {:.5f} loss: {:.5f} at iter {} ++++++'.format(
                    timestring, best_result[2], best_result[1], best_result[0]))

                if final_acc > 95:
                    save_params(exe, train_prog, os.path.join(args["model_save_dir"], save_name))


if __name__ == "__main__":
    settings = {
        "data_dir": "data/insect_cls",
        "batch_size": 64,
        "num_epochs": 300,
        "ignore_weights": ["fc6_weights", "fc6_offset"],
        "num_classes": 7,
        "l2_decay": 0.001,
        "lr": 0.0001,
        "pretrain_weights": "pretrain_weights/se_resnet50_vd",
        "model_save_dir": "models/",
        "interpolation": None,
        "resize_short_size": 128,
        "image_mean": [0.8937, 0.9031, 0.8988],
        "image_std": [0.19, 0.1995, 0.2022],
        "image_shape": [3, 112, 112],
        "lower_scale": 0.08,
        "lower_ratio": 0.75,
        "upper_ratio": 1.33,
        "save_step": 1,
        "log_iter": 50,
        "use_cuda": True,
        "_eval": True,
    }

    train_cls(settings)
