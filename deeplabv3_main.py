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
import numpy as np

# writer_train = SummaryWriter("DeepLabV3+_RS_Seg_newdataset_run5/train")
writer_train = SummaryWriter('DeepLabV3_RS_Seg_newdataset_run1/train')
# writer_val = SummaryWriter("DeepLabV3+_RS_Seg_newdataset_run5/val")
writer_val = SummaryWriter('DeepLabV3_RS_Seg_newdataset_run1/val')
# writer_all = SummaryWriter("DeepLabV3+_RS_Seg_newdataset_run5/all")
writer_all = SummaryWriter("DeepLabV3_RS_Seg_newdataset_run1/all")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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


def train():
    # net = DeepLabv3(num_classes=6, backbone='resnet101', pretrained=True).to(device)
    net = DeepLab(num_classes=6, backbone='resnet101', pretrained=True).to(device)
    net.load_state_dict(
        torch.load('models/DeepLabV3_RS_Seg_newdataset_Run_batch16_epoch18model.pth', map_location='cuda:0'))
    batch_size = 8
    num_epochs = 100
    train_iter = 0

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-4)

    rs_dataset = RsDataset(train_src_root, train_label_root, src_transform=src_transforms,
                           label_transform=label_transforms)
    dataloaders = DataLoader(rs_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    for epoch in range(num_epochs):

        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        train_iter += 1
        dt_size = len(dataloaders.dataset)
        epoch_loss = 0
        total_iou = 0
        step = 0

        net = net.train()

        for x, y in dataloaders:
            step += 1

            inputs = x.to(device)
            labels = y.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            # a = outputs
            # a = outputs.cpu().detach().numpy()
            # count = 0
            # for i in range(16):
            #    count += a[0, i, 1, 1]
            # labels = torch.squeeze(labels, 1)
            # b = labels.cpu().detach().numpy()
            # print(outputs.shape, labels.shape)
            # outputs = torch.tensor(outputs['out'], requires_grad=True)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            outputs = torch.max(outputs, 1)[1]
            iou = metrics.MIoU(outputs, labels, n_classes=6)
            total_iou += iou
            print("%d/%d, train_loss:%f, train_IoU:%f" % (step, (dt_size - 1) // dataloaders.batch_size + 1,
                                                          loss.item(), iou))
        print("epoch %d ave_loss:%f ave_IoU:%f" % (epoch + 1, epoch_loss / ((dt_size - 1) / dataloaders.batch_size + 1),
                                                   total_iou / ((dt_size - 1) / dataloaders.batch_size + 1)))

        writer_train.add_scalar("train_loss", epoch_loss / ((dt_size - 1) / dataloaders.batch_size + 1), train_iter)
        writer_train.add_scalar("train_acc", total_iou / ((dt_size - 1) / dataloaders.batch_size + 1), train_iter)

        rs_dataset_test = RsDataset(val_src_root, val_label_root, src_transform=src_transforms,
                                    label_transform=label_transforms)
        dataloaders_test = DataLoader(rs_dataset_test, batch_size=1)
        criterion = torch.nn.CrossEntropyLoss()
        net = net.eval()
        iou = test_model(net, criterion, dataloaders_test)
        print('test_IoU:', iou)
        writer_val.add_scalar("val_acc", iou, train_iter)

        # if os.path.isdir('F:/Files/RS_Seg/models'):
        #     torch.save(net.state_dict(),
        #                'F:/Files/RS_Seg/models/DeepLabV3+_RS_Seg_newdataset_Run4_batch16_epoch{}model.pth'.format(
        #                    epoch + 1))
        # else:
        #     os.makedirs('F:/Files/RS_Seg/models')
        #     torch.save(net.state_dict(),
        #                'F:/Files/RS_Seg/models/DeepLabV3+_RS_Seg_newdataset_Run4_batch16_epoch{}model.pth'.format(
        #                    epoch + 1))
        # torch.save(net.state_dict(),
        #            'F:/Files/RS_Seg/models/DeepLabV3+_RS_Seg_newdataset_Run5_batch16_epoch{}model.pth'.format(
        #                epoch + 1))
        torch.save(net.state_dict(),
                   'models/DeepLabV3_RS_Seg_newdataset_Run1_batch16_epoch{}model.pth'.format(
                       epoch + 1))


def test_model(net, criterion, dataloaders_test):
    net = net.eval()
    net = net.to(device)
    dt_size = len(dataloaders_test.dataset)
    step = 0
    epoch_loss = 0
    total_iou = 0

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
        total_iou += iou
    print("test_loss:%f" % (epoch_loss / ((dt_size - 1) / dataloaders_test.batch_size + 1)))
    writer_val.add_scalar('val_loss', (epoch_loss / ((dt_size - 1) / dataloaders_test.batch_size + 1)))
    return total_iou / ((dt_size - 1) / dataloaders_test.batch_size + 1)
