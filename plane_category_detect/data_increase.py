import cv2 
import matplotlib.pyplot as plt
import joblib
import numpy as np
import torchvision
from PIL import ImageDraw
import os
import pickle
import torch
import torchvision
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.transforms import functional as F
import cv2
import random
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import classification_report
import random
from sklearn.metrics import confusion_matrix
import selectivesearch

font = cv2.FONT_HERSHEY_SIMPLEX
import sys
import torch
os.environ['DISPLAY'] = ':1.0'



def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    # 论文里的公式2：求出B的rw,rh
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    # 论文里的公式2：求出B的rx,ry
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    # 限制坐标区域不超过样本大小
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    # 返回裁剪B区域的坐标值
    return bbx1, bby1, bbx2, bby2

def collate_fn_coco(batch):
    return tuple(zip(*batch))


def smote():
#SMOTE算法由此开始
    for i in index_minority:
        print(i)
        for epoch in range(10):
            length4 = len(dataset[i])
            # print("得到:\n",length4)
            x1 = random.randint(0,length4-1)
            ff = random.random()
            x2 = random.randint(0,length4-1)
            x1 = dataset[i][x1]
            x2 = dataset[i][x2]
            print(x1.shape)
            print(x2.shape)
            cv2.imwrite("sample_smote1.png",x1)
            cv2.imwrite("sample_smote2.png",x2)
            new_data = x1 + ff * abs(x1-x2)
            cv2.imwrite("sample_new_smote.png",new_data)
            new_data = np.array(new_data)
            dataset1[i].append(new_data)
    num = []
    for i in range(4):
        num.append(len(dataset[i]))
    plt.plot(range(4),num)
    plt.savefig("result_SMOTE.png")

def CutMix():
    #首先随机选择两个少数类
    for i in range(50):
        x1 = random.randint(0,length-1)
        x2 = random.randint(0,length-1)
        index1 = x1
        index2 = x2
        x1 = index_minority[x1]
        x2 = index_minority[x2]
        print("本轮选择的少数类是{}类和{}类".format(x1,x2))
        #接下来在这两个少数类中随机产生两个样本
        length1 = len(dataset[x1])
        length2 = len(dataset[x2])
        data1 = random.randint(0,length1-1)
        data2 = random.randint(0,length2-1)
        data1 = dataset[x1][data1]
        data2 = dataset[x2][data2]
        size = data1.shape
        # print(size)
        lam = random.random()
        x1,y1,x2,y2 = rand_bbox(size,lam)
        # print(x1,x2,y1,y2)
        # cv2.imwrite("sample1_c.png",data1)
        # cv2.imwrite("sample2_c.png",data2)
        #接下来生成随机数λ
        new_data = np.zeros((100,100,3))
        new_data = data2
        new_data[x1:x2,y1:y2] = data1[x1:x2,y1:y2]
        new_data = np.array(new_data)

        
        new_label = lam * index1 + (1-lam) * index2
        if new_label >= (index1+index2) / 2:
            new_label = index2
        else:
            new_label = index1
        dataset1[new_label].append(new_data)
    num = []
    for i in range(4):
        num.append(len(dataset[i]))
    plt.plot(range(4),num)
    plt.savefig("result_CutMix.png")
        
# cv2.imwrite("new_sample_c.png",new_data)
# dataset[1] = dataset[1][:3000]
def Mixup():
    for i in range(50):
        x1 = random.randint(0,length-1)
        x2 = random.randint(0,length-1)
        index1 = x1
        index2 = x2
        x1 = index_minority[x1]
        x2 = index_minority[x2]
        print("本轮选择的少数类是{}类和{}类".format(x1,x2))
        #接下来在这两个少数类中随机产生两个样本
        length1 = len(dataset[x1])
        length2 = len(dataset[x2])
        data1 = random.randint(0,length1-1)
        data2 = random.randint(0,length2-1)
        data1 = dataset[x1][data1]
        data2 = dataset[x2][data2]
        # cv2.imwrite("sample1.png",data1)
        # cv2.imwrite("sample2.png",data2)
        #接下来生成随机数λ
        theta = random.random()
        # print(theta)
        new_data = theta * data1 + (1-theta) * data2
        new_label = theta * index1 + (1-theta) * index2
        if new_label >= (index1+index2) / 2:
            new_label = index2
        else:
            new_label = index1
        dataset1[new_label].append(new_data)
    num = []
    for i in range(4):
        num.append(len(dataset[i]))
    plt.plot(range(4),num)
    plt.savefig("result_Mixup.png")

        
if __name__ =="__main__":
    
    
    nums = np.zeros(5)
    path = '6/'
    dataset = {}
    dataset1 = {}
    for i in range(5):
        dataset[i] = []
    for i in range(5):
        dataset1[i] = []
    index = -1
    for file_folder in os.listdir(path):
        print(file_folder)
        index += 1
        for filename in os.listdir(path+file_folder):
            path_name = path + file_folder + '/' + filename
            img = cv2.imread(path_name)
            img = cv2.resize(img,(200,300))
            dataset[index].append(img)
        
 
    index_minority = []
    index_minority.append(0)
    index_minority.append(1)
    index_minority.append(2)
    index_minority.append(3)
    index_minority.append(4)
    


    length = len(index_minority)
    smote()
    Mixup()
    CutMix()
    nu = np.zeros(5)
    for i in range(5):
        nu[i] = len(dataset[i])
    plt.plot(range(5),nu)
    plt.savefig("finall2.png")
    p1 = '6/M/'
    p2 = '6/F/'
    p3 = '6/S/'
    p4 = '6/J/'
    p5 = '6/other/'
    i1 = 0
    i2 = 0
    i3 = 0
    i4 = 0
    i5 = 0
    path = [p1, p2, p3, p4,p5]
    for i in range(5):
        for j in range(len(dataset1[i])):
            path1 = path[i] + "{}.jpg".format(j)
            # print(path1)
            cv2.imwrite(path1,dataset1[i][j])

