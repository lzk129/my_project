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
from torch.nn import functional as F

class focal_loss(nn.Module):
 
    def __init__(self, gamma=2,alpha=0.25):
        super(focal_loss, self).__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss()
        self.alpha=alpha
    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = self.alpha*(1 - p) ** self.gamma * logp * target.long() + \
               (1-self.alpha)*(p) ** self.gamma * logp * (1-target.long())
        return loss.mean()

def linear_combination(x, y, epsilon):  
    return epsilon*x + (1-epsilon)*y

def reduce_loss(loss, reduction='mean'): 
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss 

class LabelSmoothingCrossEntropy(nn.Module): 
    def __init__(self, epsilon:float=0.1, reduction='mean'): 
        super().__init__() 
        self.epsilon = epsilon 
        self.reduction = reduction 
 
    def forward(self, preds, target): 
        n = preds.size()[-1] 
        log_preds = F.log_softmax(preds, dim=-1) 
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction) 
        nll = F.nll_loss(log_preds, target, reduction=self.reduction) 
        return linear_combination(loss/n, nll, self.epsilon)

class GHMC(nn.Module):

    def __init__(self, bins=10, momentum=0, use_sigmoid=True, loss_weight=1.0,alpha=None):
        super(GHMC, self).__init__()
        self.bins = bins
        self.momentum = momentum
        edges = torch.arange(bins + 1).float() / bins
        self.register_buffer('edges', edges)
        self.edges[-1] += 1e-6
        if momentum > 0:
            acc_sum = torch.zeros(bins)
            self.register_buffer('acc_sum', acc_sum)
        self.use_sigmoid = use_sigmoid
        if not self.use_sigmoid:
            raise NotImplementedError
        self.loss_weight = loss_weight

        self.label_weight = alpha

    def forward(self, pred, target, label_weight =None, *args, **kwargs):
        target = torch.zeros(target.size(0),5).to(target.device).scatter_(1,target.view(-1,1),1)
        
        # 暂时不清楚这个label_weight输入形式，默认都为1
        if label_weight is None:
            label_weight = torch.ones([pred.size(0),pred.size(-1)]).to(target.device)

        target, label_weight = target.float(), label_weight.float()
        edges = self.edges
        mmt = self.momentum
        weights = torch.zeros_like(pred)

        # gradient length
        # sigmoid梯度计算
        g = torch.abs(pred.sigmoid().detach() - target)
        # 有效的label的位置
        valid = label_weight > 0
        # 有效的label的数量
        tot = max(valid.float().sum().item(), 1.0)
        n = 0  # n valid bins
        for i in range(self.bins):
            # 将对应的梯度值划分到对应的bin中， 0-1
            inds = (g >= edges[i]) & (g < edges[i + 1]) & valid
            # 该bin中存在多少个样本
            num_in_bin = inds.sum().item()
            if num_in_bin > 0:
                if mmt > 0:
                    # moment计算num bin
                    self.acc_sum[i] = mmt * self.acc_sum[i] \
                        + (1 - mmt) * num_in_bin
                    # 权重等于总数/num bin
                    weights[inds] = tot / self.acc_sum[i]
                else:
                    weights[inds] = tot / num_in_bin
                n += 1
        if n > 0:
            # scale系数
            weights = weights / n

        loss = F.binary_cross_entropy_with_logits(
            pred, target, weights, reduction='sum') / tot
        return loss * self.loss_weight

def fgsm_attack(image, epsilon, data_grad):
    # 使用sign（符号）函数，将对x求了偏导的梯度进行符号化
    sign_data_grad = np.sign(data_grad)
    
    # 通过epsilon生成对抗样本
    perturbed_image = data_grad + epsilon*sign_data_grad
    perturbed_image = torch.tensor(perturbed_image)
    # 做一个剪裁的工作，将torch.clamp内部大于1的数值变为1，小于0的数值等于0，防止image越界
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

class MyDataset(Dataset):
    def __init__(self,txtpath):
        #创建一个list用来储存图片和标签信息
        imgs = []
        #打开第一步创建的txt文件，按行读取，将结果以元组方式保存在imgs里
        datainfo = open(txtpath,'r')
        for line in datainfo:
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
        return pic,label





# 实例化对象
txtpath = "img_label2.txt"
data = MyDataset(txtpath)

data_loader = DataLoader(data,batch_size=20,shuffle=True,num_workers=0)


def char2num(s):
    digits = {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}
    return digits[s]

def f(x,y):
    return x * 10 + y

str = '5632'
model = torchvision.models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,5)
print("模型")

criterion = focal_loss()
param_groups=[
    {'params':model.parameters()},
    {'params':criterion.parameters()}]
optimizer = optim.SGD(param_groups, lr=0.02)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = model.to(device)
res = []
test_label = []
accuracy = []
print("准备")
for epoch in range(10):
    total = 0
    running_loss = 0.0
    runnning_correct = 0
    for img,label in data_loader:
        img = img.to(device)
        output = model(img)
        _,predicted = torch.max(output.data,1)
        print(predicted)
        label = list(map(int,label))
        print(label)
        label = torch.tensor(label,dtype=torch.long)
        label = label.to(device)
        loss = criterion(output,label)
        print_loss = loss.data.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
       
        # epoch += 1
    try:
        _,predicted = torch.max(output.data,1)
        res.append(predicted)
        test_label.append(label)
        total += label.size(0)
        runnning_correct += (predicted == label).sum()
        accuracy.append((runnning_correct/total))
        # print(x)
        print(accuracy)
        # print("第%d个epoch的识别准确率为: %d%%" %(epoch+1,(100 * runnning_correct / total)))
    except:
        print("出现问题")
    torch.save(model.state_dict(),'battle.pth')
    
    
    
