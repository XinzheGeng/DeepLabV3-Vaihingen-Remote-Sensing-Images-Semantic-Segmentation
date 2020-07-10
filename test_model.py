import time
from torch.utils.data import DataLoader
from torchvision import transforms
from path import *
import torch
from dataset import *
import numpy as np
import cv2
import os
from net.deeplabv3 import DeepLabv3
from network.deepgcnlab.deepgcnlab import DeepGCNLab

src_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


def label_transforms(x):
    # img = transforms.ToTensor()(x)
    img = torch.from_numpy(x)
    img = img.type(torch.LongTensor)
    return img


def test_model(model, dataloader):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.eval().to(device)
    num = 0
    for x, y in dataloader:
        inputs = x.to(device)
        outputs = model(inputs)
        outputs = torch.max(outputs, 1)[1]
        outputs = torch.squeeze(outputs).cpu().detach().numpy()
        outputs = Image.fromarray(np.uint8(outputs))
        outputs.save(test_ori + '/outputs/%d.png' % num)
        num += 1


"""
输入：图片路径(path+filename)，裁剪获得小图片的列数、行数（也即宽、高）
"""


def clip_one_picture(path, filename, cols, rows):
    img = cv2.imread(path + filename)  # 读取彩色图像，图像的透明度(alpha通道)被忽略，默认参数;灰度图像;读取原始图像，包括alpha通道;可以用1，0，-1来表示
    sum_rows = img.shape[0]  # 高度
    sum_cols = img.shape[1]  # 宽度
    save_path = path + "\\crop{0}_{1}\\".format(cols, rows)  # 保存的路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    print("裁剪所得{0}列图片，{1}行图片.".format(int(sum_cols / cols), int(sum_rows / rows)))

    for i in range(int(sum_cols / cols)):
        for j in range(int(sum_rows / rows)):
            cv2.imwrite(
                save_path + os.path.splitext(filename)[0] + '_' + str(j) + '_' + str(i) + '.png',
                img[j * rows:(j + 1) * rows, i * cols:(i + 1) * cols, :])
            # print(path+"\crop\\"+os.path.splitext(filename)[0]+'_'+str(j)+'_'+str(i)+os.path.splitext(filename)[1])
    print("裁剪完成，得到{0}张图片.".format(int(sum_cols / cols) * int(sum_rows / rows)))
    print("裁剪所得图片的存放地址为：{0}".format(save_path))


"""
输入：图片路径(path+filename)，裁剪所的图片的列的数量、行的数量
输出：无
"""


def merge_picture(merge_path, num_of_cols, num_of_rows):
    filename = file_name(merge_path, ".png")
    shape = cv2.imread(filename[0], 1).shape  # 三通道的影像需把-1改成1
    cols = shape[1]
    rows = shape[0]
    channels = shape[2]
    dst = np.zeros((rows * num_of_rows, cols * num_of_cols, channels), np.uint8)
    for i in range(len(filename)):
        img = cv2.imread(filename[i], -1)
        cols_th = int(filename[i].split("_")[-1].split('.')[0])  # i
        rows_th = int(filename[i].split("_")[-2].split('\\')[-1])  # j
        # cols_th = int(filename[i].split("_")[-1].split('.')[0])
        # rows_th = int(filename[i].split("_")[-2])
        roi = img[0:rows, 0:cols, :]
        dst[rows_th * rows:(rows_th + 1) * rows, cols_th * cols:(cols_th + 1) * cols, :] = roi
    cv2.imwrite(merge_path + "merge.tif", dst)
    print('Finished !')


"""遍历文件夹下某格式图片"""


def file_name(root_path, picturetype):
    filename = []
    for root, dirs, files in os.walk(root_path):
        for file in files:
            if os.path.splitext(file)[1] == picturetype:
                filename.append(os.path.join(root, file))
    return filename


"""裁剪"""
path = test_ori_labelRGB  # 要裁剪的图片所在的文件夹
filename = '/top_mosaic_09cm_area28.tif'  # 要裁剪的图片名
cols = 256  # 小图片的宽度（列数）
rows = 256  # 小图片的高度（行数）

"""合并"""
# merge_path = test_ori_labelRGB_crop  # 要合并的小图片所在的文件夹
merge_path = test_ori + '/outputs'
num_of_cols = 7  # 列数
num_of_rows = 10  # 行数

if __name__ == '__main__':
    model = DeepLabv3(num_classes=6, backbone='resnet101', pretrained=True)
    model.load_state_dict(
        torch.load('models/DeepLabV3+_RS_Seg_newdataset_Run4_batch16_epoch12model.pth', map_location='cuda:1'))
    dataset = RsDataset(test_ori_top_crop, test_ori_labelRGB_crop, src_transforms, label_transforms)
    dataloader_test = DataLoader(dataset, batch_size=1)
    start_time = time.process_time()
    test_model(model, dataloader_test)
    end_time = time.process_time()
    print('Running time: %f' % (end_time - start_time))
    os.system('python class2label.py')

    # clip_one_picture(path, filename, cols, rows)
    merge_picture(merge_path, num_of_cols, num_of_rows)

