import torchvision.models as models
import time
import torch.nn as nn
import torch.optim as op
import torch
from data_preprocess import load_data,load_data1
from torch.autograd import Variable
'''
model_ft = models.resnet18(pretrained = True)#ResNet架构
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()#损失函数
optimizer_ft = op.SGD(model_ft.parameters(), lr = learning_rate, momentum = 0.9)#SDG优化器
exp_lr_scheduler = op.lr_scheduler.StepLR(optimizer_ft, step_size = 7, gamma = 0.1)
'''
learning_rate = 0.001
class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=2,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)
        self.out = nn.Linear(10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # print(x.size())
        x = x.view(x.shape[0], -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.out(x)
        return x
'''
def train():
    train_data_gen, valid_data_gen,ceshi_data_gen = load_data()

    dataset_sizes = {'train': len(train_data_gen.dataset), 'yanzheng': len(valid_data_gen.dataset),
                     'ceshi': len(ceshi_data_gen)}
    dataloaders = {'train': train_data_gen, 'yanzheng': valid_data_gen, 'ceshi': ceshi_data_gen}
    print('train...')
    epoch_num = 25
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    optimizer = op.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # SDG优化器
    criterion = nn.CrossEntropyLoss().to(device)
    exp_lr_scheduler = op.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    exp_lr_scheduler.step()
    model.train(True)
    for epoch in range(epoch_num):
        for batch_idx, (data, target) in enumerate(train_data_gen, 0):
            data, target = Variable(data).to(device), Variable(target.long()).to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_data_gen.dataset),
                           100. * batch_idx / len(train_data_gen), loss.item()))

    torch.save(model.state_dict(), "weight_dog_cat.pt")
def train():
    since = time.time()
    train_data_gen, valid_data_gen,ceshi_data_gen = load_data()
    dataset_sizes = {'train': len(train_data_gen.dataset),
                     'yanzheng': len(valid_data_gen.dataset),
                     'ceshi': len(ceshi_data_gen.dataset)}
    dataloaders = {'train': train_data_gen, 'yanzheng': valid_data_gen, 'ceshi': ceshi_data_gen}
    print('train...')
    epoch_num = 25
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    optimizer = op.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # SDG优化器
    criterion = nn.CrossEntropyLoss().to(device)
    exp_lr_scheduler = op.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(epoch_num):
        for phase in ['train','yanzheng']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train(True)  # 模型设置为训练模式
            else:
                model.train(False)  # 模型设置为评估模式
            running_loss = 0.0
            running_correct = 0
            for data in dataloaders[phase]:
                inputs, labels = data# 获取输入
                inputs, labels = Variable(inputs), Variable(labels)# 封装成变量
                optimizer.zero_grad()# 梯度参数清零
                outputs = model(inputs)# 前向
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
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # 深度复刻模型
            if phase == 'yanzheng' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    time_elapsed = time.time() - since
    print('Training Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # 加载最优权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), "weight_dog_cat.pt")
'''
def test():
    train_data_gen, valid_data_gen,ceshi_data_gen = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load(r"F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/weights/weight_dog_cat.pt"), False)
    model.eval()
    total = 0
    current = 0
    for data in ceshi_data_gen:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
    for data in train_data_gen:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
    for data in valid_data_gen:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()

    print('Accuracy:%d%%' % (100 * current / total))

def train1():
    since = time.time()

    train_data_gen1, valid_data_gen1,ceshi_data_gen1,\
    train_data_gen2, valid_data_gen2,ceshi_data_gen2,\
    train_data_gen3, valid_data_gen3,ceshi_data_gen3 = load_data1()
    dataset_sizes = {'train': len(train_data_gen1.dataset)+len(train_data_gen2.dataset)+len(train_data_gen3.dataset),
                     'yanzheng': len(valid_data_gen1.dataset)+len(valid_data_gen2.dataset)+len(valid_data_gen3.dataset),
                     'ceshi': len(ceshi_data_gen1.dataset)+len(ceshi_data_gen2.dataset)+len(ceshi_data_gen3.dataset)}
    dataloaders = {'train1': train_data_gen1, 'yanzheng1': valid_data_gen1, 'ceshi1': ceshi_data_gen1,
                   'train2': train_data_gen2, 'yanzheng2': valid_data_gen2, 'ceshi2': ceshi_data_gen2,
                   'train3': train_data_gen3, 'yanzheng3': valid_data_gen3, 'ceshi3': ceshi_data_gen3}
    print('train...')
    epoch_num = 20
    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    optimizer = op.SGD(model.parameters(), lr=learning_rate, momentum=0.9)  # SDG优化器
    criterion = nn.CrossEntropyLoss().to(device)
    exp_lr_scheduler = op.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    best_model_wts = model.state_dict()
    best_acc = 0.0
    for epoch in range(epoch_num):
        for phase in ['train', 'yanzheng']:
            if phase == 'train':
                exp_lr_scheduler.step()
                model.train(True)  # 模型设置为训练模式
            else:
                model.train(False)  # 模型设置为评估模式
            running_loss = 0.0
            running_correct = 0
            if phase=='train':
                for phases in ['train1','train2','train3']:
                    for data in dataloaders[phases]:
                        inputs, labels = data  # 获取输入
                        inputs, labels = Variable(inputs), Variable(labels)  # 封装成变量
                        optimizer.zero_grad()  # 梯度参数清零
                        outputs = model(inputs)  # 前向
                        _, preds = torch.max(outputs.data, 1)
                        loss = criterion(outputs, labels)
                        # 只在训练阶段反向优化
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                        # 统计
                        running_loss += loss.item()
                        running_correct += torch.sum(preds == labels.data)
            elif phase=='yanzheng':
                for phases in ['yanzheng1','yanzheng2','yanzheng3']:
                    for data in dataloaders[phases]:
                        inputs, labels = data  # 获取输入
                        inputs, labels = Variable(inputs), Variable(labels)  # 封装成变量
                        optimizer.zero_grad()  # 梯度参数清零
                        outputs = model(inputs)  # 前向
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
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            # 深度复刻模型
            if phase == 'yanzheng' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        print()
    time_elapsed = time.time() - since
    print('Training Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # 加载最优权重
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), r"F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/weights/weight_dog_cat.pt")
def test1():
    train_data_gen1, valid_data_gen1, ceshi_data_gen1, \
    train_data_gen2, valid_data_gen2, ceshi_data_gen2, \
    train_data_gen3, valid_data_gen3, ceshi_data_gen3 = load_data1()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load(r"F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/weights/weight_dog_cat.pt"), False)
    model.eval()
    total = 0
    current = 0
    for data in ceshi_data_gen1:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
    for data in ceshi_data_gen2:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()
    for data in ceshi_data_gen3:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        total += labels.size(0)
        current += (predicted == labels).sum()

    print('Accuracy:%d%%' % (100 * current / total))