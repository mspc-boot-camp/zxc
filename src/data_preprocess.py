import torchvision.datasets as dset
import torchvision.transforms as ts
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.autograd import Variable
import torch
path=r'F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/input/'
def load_data():
    print('Data processing...')
    simple_transform = ts.Compose([ ts.Resize((224, 224)),
                                    ts.ToTensor(),
                                    ts.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ])
    train = dset.ImageFolder(path + r'training/', simple_transform)
    valid = dset.ImageFolder(path + r'yanzheng/', simple_transform)
    ceshi = dset.ImageFolder(path + r'ceshiji/', simple_transform)
    train_data_gen = torch.utils.data.DataLoader(train, batch_size=10, shuffle=True, num_workers=0)
    valid_data_gen = torch.utils.data.DataLoader(valid, batch_size=1, shuffle=True, num_workers=0)
    ceshi_data_gen = torch.utils.data.DataLoader(ceshi, batch_size=1, shuffle=True, num_workers=0)
    return train_data_gen,valid_data_gen,ceshi_data_gen
def load_data1():
    print('Data processing...')
    simple_transform1 = ts.Compose([ts.Resize((224, 224)),
                                   ts.ToTensor(),
                                   ts.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    train1 = dset.ImageFolder(path+r'training/', simple_transform1)
    valid1 = dset.ImageFolder(path+r'yanzheng/', simple_transform1)
    ceshi1 =dset.ImageFolder(path+r'ceshiji/', simple_transform1)
    simple_transform2 = ts.Compose([ts.RandomHorizontalFlip(p=1),
                                   ts.Resize((224, 224)),
                                   ts.ToTensor(),
                                   ts.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    train2 = dset.ImageFolder(path + r'training/', simple_transform2)
    valid2 = dset.ImageFolder(path + r'yanzheng/', simple_transform2)
    ceshi2 = dset.ImageFolder(path + r'ceshiji/', simple_transform2)
    simple_transform3 = ts.Compose([ts.RandomVerticalFlip(p=1),
                                   ts.Resize((224, 224)),
                                   ts.ToTensor(),
                                   ts.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    train3 = dset.ImageFolder(path + r'training/', simple_transform3)
    valid3 = dset.ImageFolder(path + r'yanzheng/', simple_transform3)
    ceshi3 = dset.ImageFolder(path + r'ceshiji/', simple_transform3)
    # 批量加载
    train_data_gen1 = torch.utils.data.DataLoader(train1,  batch_size=10, shuffle=True, num_workers=0)
    valid_data_gen1 = torch.utils.data.DataLoader(valid1,  batch_size=1, shuffle=True, num_workers=0)
    ceshi_data_gen1 = torch.utils.data.DataLoader(ceshi1, batch_size=1, shuffle=True, num_workers=0)

    train_data_gen2 = torch.utils.data.DataLoader(train2, batch_size=10, shuffle=True, num_workers=0)
    valid_data_gen2 = torch.utils.data.DataLoader(valid2, batch_size=1, shuffle=True, num_workers=0)
    ceshi_data_gen2 = torch.utils.data.DataLoader(ceshi2, batch_size=1, shuffle=True, num_workers=0)

    train_data_gen3 = torch.utils.data.DataLoader(train3, batch_size=10, shuffle=True, num_workers=0)
    valid_data_gen3 = torch.utils.data.DataLoader(valid3, batch_size=1, shuffle=True, num_workers=0)
    ceshi_data_gen3 = torch.utils.data.DataLoader(ceshi3, batch_size=1, shuffle=True, num_workers=0)

    return train_data_gen1,valid_data_gen1,ceshi_data_gen1,train_data_gen2,valid_data_gen2,ceshi_data_gen2,train_data_gen3,valid_data_gen3,ceshi_data_gen3





'''
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # 每轮都有训练和验证的阶段
        for phase in ['train', 'valid','ceshi']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # 模型设置为训练模式
            else:
                model.train(False)  # 模型设置为评估模式

            running_loss = 0.0
            running_correct = 0

            # 在数据上迭代
            for data in dataloaders[phase]:
                # 获取输入
                inputs, labels = data
                # 封装成变量
                inputs, labels = Variable(inputs), Variable(labels)
                # 梯度参数清零
                optimizer.zero_grad()
                # 前向
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)
                # 只在训练阶段反向优化
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # 统计
                running_loss += loss.item()
                running_correct += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]

            print('{} Loss: (:.4f) Acc: (:.4f)'.format(phase, epoch_loss, epoch_acc))

            # 深度复刻模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    time_elapsed = time.time() - since
    print('Training Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # 加载最优权重
    model.load_state_dict(best_model_wts)
    return model


def data_chuli():
    path_lines=read_picture()
    chuli_picture=[]
    for lujing in path_lines:
        image=cv2.imread(lujing)
        image_suofang = cv2.resize(image, (200, 200))  # shrink picture
        chuli_picture.append(image_suofang)
'''