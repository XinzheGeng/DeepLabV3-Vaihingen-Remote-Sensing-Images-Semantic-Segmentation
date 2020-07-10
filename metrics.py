import numpy as np
import torch


def MIoU(prediction, target, n_classes):
    prediction = prediction.flatten()
    '''
    flatten() 默认按行方向降维
    array(  [[1, 2],
            [3, 4],
            [5, 6]])
    output : array([1, 2, 3, 4, 5, 6])
    '''
    target = target.flatten()
    ious = []
    for cls in range(1, n_classes):
        pred = prediction == cls
        tar = target == cls
        delta = 1e-10
        IoU = ((pred * tar).sum() + delta) / (pred.sum() + tar.sum() - (pred * tar).sum() + delta)
        ious.append(IoU)

    return torch.mean(torch.stack(ious))


def IoU(prediction, target, n_classes):
    prediction = prediction.flatten()
    target = target.flatten()
    ious = []
    for cls in range(1, n_classes):
        pred = prediction == cls
        tar = target == cls
        delta = 1e-10
        IoU = ((pred * tar).sum() + delta) / (pred.sum() + tar.sum() - (pred * tar).sum() + delta)
        ious.append(IoU)

    return ious

