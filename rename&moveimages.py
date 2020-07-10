import uuid
import os, random, shutil
from path import *

path = test_ori + '/outputs'
startNum = 0
count = 0
# name = 'YRDQ'
fileType = '.png'

filelist = os.listdir(path)

for files in sorted(filelist, key=lambda x: int(x[:-4])):
    olddir = os.path.join(path, files)
    if os.path.isdir(olddir):
        continue
    # newdir = os.path.join(path, str(count + int(startNum)) + fileType)
    newdir = os.path.join(path, str(int(count / 7)) + '_' + str(count % 7) + fileType)
    # uuid_str = uuid.uuid4().hex
    # os.rename(olddir, os.path.join(path, uuid_str) + fileType)
    os.rename(olddir, newdir)
    count += 1
print('Rename complete !')

'''
# random.seed(1)
srcDir = "F:/Files/RS_Seg/datasets/raw/Images/"  # 源图片文件夹路径
labelDir = "F:/Files/RS_Seg/datasets/label/Images/"

tar_srcDir = 'F:/Files/RS_Seg/datasets/val_420/src/'  # 移动到新的文件夹路径
tar_labelDir = 'F:/Files/RS_Seg/datasets/val_420/label/'

path_srcDir = os.listdir(srcDir)  # 取图片的原始路径
path_labelDir = os.listdir(labelDir)
# print(len(path_srcDir), path_labelDir)
# filenumber = len(path_srcDir)
# rate = 0.8  # 自定义抽取图片的比例
# picknumber = int(filenumber * rate)  # 按照rate比例从文件夹中取一定数量图片
# src_sample = random.sample(path_srcDir, picknumber)  # 随机选取picknumber数量的样本图片
# label_sample = random.sample(path_labelDir, picknumber)

src_list = []
label_list = []
src_sample = []
label_sample = []

for i in range(0, 2099, 5):
    # src_sample.append(path_srcDir[i])
    label_sample.append(path_labelDir[i])

for name in src_sample:
    shutil.move(srcDir + name, tar_srcDir + name)

for name in label_sample:
    shutil.move(labelDir + name, tar_labelDir + name)
print('Move complete !')
'''
