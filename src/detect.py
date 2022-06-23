import torchvision.transforms as ts
from build_network import test,train1,cnn,test1
import torchvision.datasets as dset
import torch
from PIL import Image
import time
import csv
from torch.utils.data import Dataset, DataLoader

path=r'F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/input/'

class MyDataset(Dataset):
    def __init__(self, data, transform, loder):
        self.data = data
        self.transform = transform
        self.loader = loder
    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.data)
def find_label(str):
    first, last = 0, 0
    for i in range(len(str) - 1, -1, -1):
        if str[i - 1] == '-':
            last = i - 1
        if (str[i] == 'c' or str[i] == 'd') and str[i - 1] == '/':
            first = i
            break

    name = str[first:last]
    if name == 'dog':
        return 1
    else:
        return 0
def Myloader(path):
    return Image.open(path).convert('RGB')
def fenlei():
    image_path=[]
    with open(path+r'image.txt') as f:
        path_lines = f.readlines()
        for i in path_lines:
            i=i.replace('\n', '')
            i = i.replace('\\', '/')
            name=find_label(i)
            image_path.append([i,name])
    simple_transform = ts.Compose([ts.Resize((224, 224)),
                                   ts.ToTensor(),
                                   ts.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = cnn().to(device)
    model.load_state_dict(torch.load("weight_dog_cat.pt"), False)
    model.eval()
    datas=MyDataset(image_path, transform=simple_transform, loder=Myloader)
    data_gen=DataLoader(dataset=datas, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    f=open('F:/Yanjiusheng/Yan1_shangxueqi/New_student_practice/work6/dog_cat_class2/output/out_results.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["结果", "时间"])
    for data in data_gen:
        since = time.time()
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = torch.max(outputs.data, 1)[1].data
        time_elapsed = time.time() - since
        if int(predicted)==0:
            csv_writer.writerow(['猫',str(time_elapsed)])
        elif int(predicted)==1:
            csv_writer.writerow(['狗',str(time_elapsed)])
    f.close()

if __name__ == '__main__':
    #train1()
    test1()
    #fenlei()



