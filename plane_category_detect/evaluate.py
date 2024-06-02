import os
import torch
import torch.nn as nn
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from functools import reduce
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from timeit import default_timer as timer
import argparse
tic = timer()
# 待测试的代码




class MyDataset(Dataset):
    def __init__(self,txtpath):
        #创建一个list用来储存图片和标签信息
        imgs = []
        #打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        datainfo = open(txtpath,'r')
        i = 0
        for line in datainfo:
            print(i)
            i += 1
            line = line.strip('\n')
            words = line.split()
            imgs.append((words[0],words[1]))

        self.imgs = imgs
	#返回数据集大小
    def __len__(self):
        return len(self.imgs)
	#打开index对应图片进行预处理后return回处理后的图片和标签
    def __getitem__(self, index):
        pic,label = self.imgs[index]
        pic = Image.open(pic)
        pic = np.array(pic)
        pic = cv2.resize(pic,(200,300))
       
        pic = transforms.ToTensor()(pic)
        # pic = pic.type(torch.FloatTensor)
        return pic,label


#实例化对象
txtpath = "test_label.txt"
data = MyDataset(txtpath)
#将数据集导入DataLoader，进行shuffle以及选取batch_size
data_loader = DataLoader(data,batch_size=176,shuffle=True,num_workers=0)

#Windows里num_works只能为0，其他值会报错
def char2num(s):
    digits = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    return digits[s]

def f(x,y):
    return x * 10 + y

str = '5632'






model = torchvision.models.resnet18(pretrained=True)
model.eval()
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,4)
print("模型")

criterion = torch.nn.CrossEntropyLoss()
param_groups=[
    {'params':model.parameters()},
    {'params':criterion.parameters()}]
optimizer = optim.SGD(param_groups, lr=0.02)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("model_best.pth"))
model = model.to(device)
res = []
test_label = []
accuracy = []
total = 0
running_loss = 0.0
runnning_correct = 0
print("准备")
for img,label in data_loader:
    img = img.to(device)
    output = model(img)
    label = list(map(int,label))
    label = torch.tensor(label,dtype=torch.long)
    label = label.to(device)
    print(label)
try:
    _,predicted = torch.max(output.data,1)
    print(predicted)
    total += label.size(0)
    runnning_correct += (predicted == label).sum()
    accuracy.append((runnning_correct/total))
    print(accuracy)
except:
    print("有问题,你不对劲")

toc = timer()
print(toc-tic)

print("总时间为{}".format(toc - tic))
    
   
