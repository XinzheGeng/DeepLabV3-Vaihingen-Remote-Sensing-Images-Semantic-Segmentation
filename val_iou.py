import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import metrics
from dataset import RsDataset
from net.deeplabv3 import DeepLabv3
from path import *

src_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def label_transforms(x):
    # img = transforms.ToTensor()(x)
    img = torch.from_numpy(x)
    img = img.type(torch.LongTensor)
    return img


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def test_model(net, dataloaders_test):
    net = net.eval()
    net = net.to(device)
    step = 0
    total_iou = [0, 0, 0, 0, 0]
    miou = []

    print('Testing...')
    for x, y in dataloaders_test:
        step += 1
        inputs = x.to(device)
        labels = y.to(device)
        outputs = net(inputs)
        outputs = torch.max(outputs, 1)[1]
        ious = metrics.IoU(outputs, labels, n_classes=6)
        for i in range(0, 5):
            total_iou[i] += ious[i]

    for i in range(0, 5):
        miou.append(total_iou[i] / len(dataloaders_test.dataset))

    return miou


if __name__ == '__main__':
    model = DeepLabv3(num_classes=6, backbone='resnet101', pretrained=True)
    model.load_state_dict(
        torch.load('models/DeepLabV3+_RS_Seg_newdataset_Run4_batch16_epoch12model.pth', map_location='cuda:0'))
    dataset = RsDataset(val_src_root, val_label_root, src_transforms, label_transforms)
    dataloader_test = DataLoader(dataset, batch_size=1)
    miou = []
    miou = test_model(model, dataloader_test)
    print(miou)
