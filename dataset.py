import torch.utils.data as data
import PIL.Image as Image
import os
import numpy as np


def make_dataset(raw, label):
    imgs = []

    # n = len(os.listdir(raw))
    # for i in range(n):
    #     img = os.path.join(raw, "%d.png" % i)
    #     mask = os.path.join(label, "%d.png" % i)
    #     imgs.append((img, mask))
    for file in os.listdir(raw):
        img = os.path.join(raw, file)
        mask = os.path.join(label, file)
        imgs.append((img, mask))
    return imgs


class RsDataset(data.Dataset):
    def __init__(self, raw, label, src_transform=None, label_transform=None):
        imgs = make_dataset(raw, label)
        self.imgs = imgs
        self.src_transform = src_transform
        self.label_transform = label_transform

        self.mapping = {
            0: (255, 255, 255),  # 背景
            1: (0, 0, 255),  # 建筑
            2: (0, 255, 255),  # 低矮植被
            3: (0, 255, 0),  # 林木
            4: (255, 255, 0),  # 汽车
            5: (255, 0, 0)  # 不透水地面
        }

    def label_to_class(self, label):
        mask = label
        mask = mask.convert('L')

        img_label = label.load()
        img_mask = mask.load()
        h, w = label.size

        for i in range(h):
            for j in range(w):
                flag = False
                for k in self.mapping:
                    if img_label[i, j] == self.mapping[k]:
                        img_mask[i, j] = k
                        flag = True
                if not flag:
                    img_mask[i, j] = 0
        return mask

    # def class_to_label(self, mask):
    #     label = mask.convert('RGB')
    #
    #     img_label = label.load()
    #     img_mask = mask.load()
    #     h, w = label.size
    #
    #     for i in range(h):
    #         for j in range(w):
    #             for k in self.mapping:
    #                 if img_mask[i, j] == k:
    #                     img_label[i, j] = self.mapping[k]
    #     return label

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        # 单通道处理
        # img_y = self.label_to_class(img_y)
        # test
        img_y = np.array(img_y)

        if self.src_transform is not None:
            img_x = self.src_transform(img_x)
        if self.label_transform is not None:
            img_y = self.label_transform(img_y)

        return img_x, img_y

    def __len__(self):
        return len(self.imgs)
