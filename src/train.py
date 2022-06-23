import os
import numpy as np
from img_read import read_picture
#from data_preprocess import train_model

path=r'F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/input/'
def train_jihe():
    files = read_picture()
    no_of_images = len(files)#创建验证集的随机文件索引
    shuffle = np.random.permutation(no_of_images)
    yanzheng_index = shuffle[:60]# 将训练数据集进行划分，60个样本归入验证集，480个样本归入训练集,剩下的归入测试集
    train_index = shuffle[60:540]
    ceshi_index=shuffle[540:]
    # 创建验证集和训练集文件夹
    os.mkdir(os.path.join(path, 'yanzheng'))
    os.mkdir(os.path.join(path, 'training'))
    os.mkdir(os.path.join(path, 'ceshiji'))
    for t in ['training', 'yanzheng','ceshiji']:
       for folder in ['dog', 'cat']:
           os.mkdir(os.path.join(path, t, folder))
    # 将图片的一小部分复制到yanzheng文件夹
    print('开始了')
    for i in yanzheng_index:
        folder = files[i].split('\\')[-1].split('-')[0]
        image=files[i].split('\\')[-1].split('\n')[:-1]
        os.rename(os.path.join(path,folder+r'_g/'+image[0]), os.path.join(path, r'yanzheng/'+folder+r'/'+image[0]))
    # 将剩下的图片复制到training文件夹
    for i in train_index:
        folder = files[i].split('\\')[-1].split('-')[0]
        image = files[i].split('\\')[-1].split('\n')[:-1]
        os.rename(os.path.join(path,folder+r'_g/'+image[0]), os.path.join(path, r'training/'+folder+r'/'+image[0]))
    for i in ceshi_index:
        folder = files[i].split('\\')[-1].split('-')[0]
        image = files[i].split('\\')[-1].split('\n')[:-1]
        os.rename(os.path.join(path,folder+r'_g/'+image[0]), os.path.join(path, r'ceshiji/'+folder+r'/'+image[0]))


#train_jihe()