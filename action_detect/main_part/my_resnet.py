import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os, PIL
import numpy as np
from torch.utils.data import DataLoader,Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


from torchvision.models import resnet50
import copy

# 构建模型
class resnet_50(nn.Module):
    def __init__(self):
        super(resnet_50, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(BottleBlock, [64, 64], 3)
        self.layer2 = self.make_layer(BottleBlock, [256, 128], 4)
        self.layer3 = self.make_layer(BottleBlock, [512, 256], 6)
        self.layer4 = self.make_layer(BottleBlock, [1024, 512], 3)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 4))

    # 创建一个拼接的模块
    def make_layer(self, module, filters, n_layer):
        filter1, filter2 = filters
        layers = nn.Sequential()
        layers.add_module('0', module(filter1, filter2))

        filter1 = filter2 * 4
        for i in range(1, n_layer):
            layers.add_module(str(i), module(filter1, filter2))
        return layers

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool1(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        return self.fc(out)


class BottleBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.ac = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(out_c, out_c * 4, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(out_c * 4)

        if in_c != out_c * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c * 4, kernel_size=1),
                nn.BatchNorm2d(out_c * 4)
            )

        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ac(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ac(out)
        out = self.conv3(out)
        out = self.bn3(out)
        x = self.downsample(x)

        return self.ac(x + out)


# 数据导入

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# path = "D:/Desktop/lzk_project/data/run"
# f = []
# for root, dirs, files in os.walk(path):
#     for name in files:
#         f.append(os.path.join(root, name))
# print("图片总数：",len(f))


path1 = 'F:/pri/lzk_project/data'
transform = transforms.Compose([
    transforms.Resize((352,352)), #统一图片大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #标准化
])
data = ImageFolder(path1, transform = transform)

print(data.class_to_idx)
print(data.classes)


# 训练
def train_model(epoch, model, optim, loss_function, train_dl, test_dl=None, lr_scheduler=None):
    for k in range(epoch):
        model.train()
        train_acc, train_len = 0, 0
        history = {"train_loss": [],
                   "train_acc": [],
                   "val_loss": [],
                   "val_acc": [],
                   'best_val_acc': (0, 0)}
        best_acc = 0.0
        for step, (x, y) in enumerate(train_dl):
            # print(step)
            x, y = x.to(device), y.to(device)
            y_h = model(x)
            optim.zero_grad()
            loss = loss_function(y_h, y)
            loss.backward()
            optim.step()
            predict = torch.max(y_h, dim=1)[1]
            train_acc += (predict == y).sum().item()
            train_len += len(y)
            trainacc = train_acc / train_len * 100
            history['train_loss'].append(loss)
            history['train_acc'].append(trainacc)

        val_loss = 0.0
        val_acc, val_len = 0, 0
        if test_dl is not None:
            model.eval()
            with torch.no_grad():
                for _, (x, y) in enumerate(test_dl):
                    x, y = x.to(device), y.to(device)
                    y_h = model(x)
                    loss = loss_function(y_h, y)
                    val_loss += loss.item()
                    predict = torch.max(y_h, dim=1)[1]
                    val_acc += (predict == y).sum().item()
                    val_len += len(y)
            acc = val_acc / val_len
            history['val_loss'].append(val_loss)
            history['val_acc'].append(acc)
        if (k + 1) % 5 == 0:
            print(
                f"epoch: {k + 1}/{epoch}\n   train_loss: {history['train_loss'][-1]},\n   train_acc: {trainacc},\n   val_acc: {acc},\n   val_loss: {history['val_loss'][-1]}")
        if lr_scheduler is not None:
            lr_scheduler.step()
        if best_acc < trainacc:
            best_acc = acc
            history['best val acc'] = (epoch, acc)
            best_model_wts = copy.deepcopy(model.state_dict())
    print('=' * 80)
    print(f'Best val acc: {best_acc}')
    return model, history, best_model_wts

# print(data)


torch.manual_seed(123)
train_ds, test_ds = torch.utils.data.random_split(data, [800,301])   #800个训练数据，251个测试数据
train_dl = DataLoader(train_ds, batch_size = 4, shuffle=True, prefetch_factor=2) #prefetch_factor 加速训练， shffule 打乱数据
test_dl = DataLoader(train_ds, batch_size = 4, shuffle=False, prefetch_factor=2 )

if __name__ == '__main__':
    model = resnet_50()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optims = torch.optim.Adam(model.parameters(), lr = 1e-2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optims, 50, last_epoch=-1)
    model, history, best_model_wts = train_model(50, model, optims, loss_fn, train_dl, test_dl,lr_scheduler)
    # 保存模型
    torch.save(best_model_wts, 'my_resnet.pth')


