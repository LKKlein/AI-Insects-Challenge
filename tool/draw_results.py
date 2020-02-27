# -*- coding: utf-8 -*-

# 此文件中定义画图相关的函数

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

from reader.insects_reader import INSECT_NAMES


def draw_rectangle(currentAxis, bbox, edgecolor='k', facecolor='y', fill=False, linestyle='-'):
    # 定义画矩形框的函数 
    # currentAxis，坐标轴，通过plt.gca()获取
    # bbox，边界框，包含四个数值的list， [x1, y1, x2, y2]
    # edgecolor，边框线条颜色
    # facecolor，填充颜色
    # fill, 是否填充
    # linestype，边框线型
    # patches.Rectangle需要传入左上角坐标、矩形区域的宽度、高度等参数
    rect=patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0]+1, bbox[3]-bbox[1]+1, linewidth=1,
                           edgecolor=edgecolor,facecolor=facecolor,fill=fill, linestyle=linestyle)
    currentAxis.add_patch(rect)


def draw_results(result, im, draw_thresh=0.5):
    # 定义绘制预测结果的函数
    plt.figure(figsize=(10, 10))
#     im = imread(filename)
    plt.imshow(im)
    currentAxis = plt.gca()
    colors = ['r', 'g', 'b', 'k', 'y', 'pink', 'purple']
    for item in result:
        box = item[2:6]
        label = int(item[0])
        name = INSECT_NAMES[label]
        if item[1] > draw_thresh:
            draw_rectangle(currentAxis, box, edgecolor=colors[label])
            plt.text(box[0], box[1], name, fontsize=12, color=colors[label])
#     plt.savefig('output_pic.png')
    plt.show()
