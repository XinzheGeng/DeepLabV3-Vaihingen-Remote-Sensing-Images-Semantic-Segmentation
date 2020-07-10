import logging
import argparse
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, models
from path import *
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch import optim
from torch.autograd import Variable
from dataset import RsDataset
import os
import metrics
from net.deeplabv3 import DeepLabv3
from net.deeplab import DeepLab
from network.deepgcnlab.deepgcnlab import DeepGCNLab
import numpy as np
import nni

# logger = logging.getLogger('RSSeg_AutoML')

# writer_train = SummaryWriter("DeepLabV3+_RS_Seg_run/train")
# writer_train = SummaryWriter('DeepLabV3_RS_Seg_newdataset_run/train')
# writer_val = SummaryWriter("DeepLabV3+_RS_Seg_run/val")
# writer_val = SummaryWriter('DeepLabV3_RS_Seg_newdataset_run/val')
# writer_all = SummaryWriter("DeepLabV3+_RS_Seg_run/all")
# writer_all = SummaryWriter("DeepLabV3_RS_Seg_newdataset_run/all")

writer_train = SummaryWriter("DeepGCNLab_RS_Seg_run1/train")
writer_val = SummaryWriter("DeepGCNLab_RS_Seg_run1/val")
writer_all = SummaryWriter("DeepGCNLab_RS_Seg_run1/all")

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]
print('CUDA: ', torch.cuda.is_available())

src_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def label_transforms(x):
    # img = transforms.ToTensor()(x)
    img = torch.from_numpy(x)
    img = img.type(torch.LongTensor)
    return img


def train(args):
    # net = DeepLabv3(num_classes=6, backbone='resnet101', pretrained=True).to(device)
    # net = DeepLab(num_classes=6, backbone='resnet101', pretrained=True).to(device)
    net = DeepGCNLab(num_classes=6, backbone='resnet101', pretrained=False)
    net = torch.nn.DataParallel(net, device_ids=device_ids)
    net = net.cuda(device=device_ids[0])
    # net.load_state_dict(
    #     torch.load('models/DeepGCNLab_RS_Seg_Run_batch8_epoch5model.pth', map_location='cuda:1'))

    start_epoch = 0
    total_epochs = 500
    train_iter = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), args['lr'])

    rs_dataset = RsDataset(train_src_root, train_label_root, src_transform=src_transforms,
                           label_transform=label_transforms)
    dataloaders = DataLoader(rs_dataset, batch_size=args['batch_size'], shuffle=False, num_workers=0)

    for epoch in range(start_epoch, total_epochs):

        print('Epoch {}/{}'.format(epoch + 1, total_epochs))
        print('-' * 10)
        train_iter += 1
        dt_size = len(dataloaders.dataset)
        epoch_loss = 0
        epoch_iou = 0
        step = 0
        total_step = (dt_size - 1) // dataloaders.batch_size + 1

        net = net.train()

        for x, y in dataloaders:
            step += 1

            inputs = x.cuda(device=device_ids[0])
            labels = y.cuda(device=device_ids[0])

            optimizer.zero_grad()

            outputs = net(inputs)
            # outputs = torch.tensor(outputs['out'], requires_grad=True)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            outputs = torch.max(outputs, 1)[1]
            iou = metrics.MIoU(outputs, labels, n_classes=6)
            epoch_iou += iou

            print("%d/%d, train_loss:%f, train_IoU:%f" % (step, total_step, loss.item(), iou))
        print("epoch %d ave_loss:%f ave_IoU:%f" % (epoch + 1, epoch_loss / total_step,
                                                   epoch_iou / total_step))
        writer_train.add_scalar("train_loss", epoch_loss / total_step, train_iter)
        writer_train.add_scalar("train_acc", epoch_iou / total_step, train_iter)

        rs_dataset_test = RsDataset(val_src_root, val_label_root, src_transform=src_transforms,
                                    label_transform=label_transforms)
        dataloaders_test = DataLoader(rs_dataset_test, batch_size=1)
        criterion = torch.nn.CrossEntropyLoss()
        net = net.eval()
        iou = test_model(net, criterion, dataloaders_test, train_iter)
        print('test_IoU:', iou)
        writer_val.add_scalar("val_acc", iou, train_iter)

        # report intermediate result
        # nni.report_intermediate_result(iou)
        # logger.debug('test accuracy %g', iou)
        # logger.debug('Pipe send intermediate result done.')

        torch.save(net.state_dict(),
                   'models/DeepGCNLab_RS_Seg_Run_batch8_epoch{}model.pth'.format(
                       epoch + 1))

    # report final result
    # nni.report_final_result(iou)
    # logger.debug('Final result is %g', iou)
    # logger.debug('Send final result done.')


def test_model(net, criterion, dataloaders_test, train_iter):
    net = net.eval()
    net = net.to(device)
    dt_size = len(dataloaders_test.dataset)
    step = 0
    total_step = (dt_size - 1) // dataloaders_test.batch_size + 1
    epoch_loss = 0
    epoch_iou = 0

    print('Testing...')
    for x, y in dataloaders_test:
        step += 1

        inputs = x.to(device)
        labels = y.to(device)
        outputs = net(inputs)

        # outputs = torch.tensor(outputs['out'], requires_grad=True)

        loss = criterion(outputs, labels)
        epoch_loss += float(loss.item())
        outputs = torch.max(outputs, 1)[1]
        iou = metrics.MIoU(outputs, labels, n_classes=6)
        epoch_iou += iou
    print("test_loss:%f" % (epoch_loss / total_step))
    writer_val.add_scalar('val_loss', (epoch_loss / total_step), train_iter)
    return epoch_iou / total_step
