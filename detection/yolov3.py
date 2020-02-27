from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class YOLOv3():

    def __init__(self, backbone, head, freeze_backbone=False):
        self.freeze_backbone = freeze_backbone
        self.backbone = backbone
        self.yolo_head = head

    def __call__(self, feed_vars, mode="train"):
        inputs = feed_vars["image"]
        backbone = self.backbone(inputs)

        if self.freeze_backbone:
            for feat in backbone:
                feat.stop_gradient = True

        if mode == "train":
            gt_box = feed_vars['gt_bbox']
            gt_label = feed_vars['gt_class']
            gt_score = feed_vars['gt_score']
            result = self.yolo_head._get_outputs(backbone, is_train=True)
            return {"loss": self.yolo_head.get_loss(result, gt_box, gt_label, gt_score)}
        else:
            im_size = feed_vars["im_size"]
            result = self.yolo_head._get_outputs(backbone, is_train=False)
            pred = self.yolo_head.get_prediction(result, im_size=im_size)
        return pred

    def train(self, feed_vars):
        return self.__call__(feed_vars, mode="train")

    def eval(self, feed_vars):
        return self.__call__(feed_vars, mode="eval")

    def test(self, feed_vars):
        return self.__call__(feed_vars, mode="test")
