import torch
from PIL import Image
from torchvision import transforms
import os
from path import *
import numpy as np


def class2label(test_root, filename, save_path):
    mask = Image.open(test_root + '/' + filename)
    label = mask.convert('RGB')
    img_label = label.load()
    img_mask = mask.load()
    h, w = label.size
    mapping = {
        0: (255, 255, 255),  # 背景
        1: (0, 0, 255),  # 建筑
        2: (0, 255, 255),  # 低矮植被
        3: (0, 255, 0),  # 林木
        4: (255, 255, 0),  # 汽车
        5: (255, 0, 0)  # 不透水地面
    }

    for i in range(h):
        for j in range(w):
            for k in mapping:
                if img_mask[i, j] == k:
                    img_label[i, j] = mapping[k]
    label.save(save_path + '/' + filename)


if __name__ == '__main__':
    path_outputs = test_ori + '/outputs'
    for filename in os.listdir(path_outputs):
        # class2label(path_label, num, test_labelRGB)
        class2label(path_outputs, filename, test_ori + '/outputs')
    os.system('python "rename&moveimages.py"')
    print('Class to Label: Complete ！')
