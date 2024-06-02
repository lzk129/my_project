# -*- encoding: utf-8 -*-
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#@File    :   mcu_image_vision_server.py
#@Time    :   2021/09/02 19:48:29
#@Author  :   xianqin.ma
#@Email   :   xianqin.ma@wenge.com
#@Version :   v3.0
#@Desc    :   None
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from cProfile import label
import os
import cv2
import csv
import uuid
import json
import time

# from matplotlib import transforms
import torch
import logging
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from flask import Flask
from flask import request
from flasgger import Swagger
from flasgger.utils import swag_from
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import torchvision
import torch.optim as optim
import torch.nn as nn


import mxnet as mx
import torch.nn.functional as F
from torch.utils.data import Dataset


app = Flask(__name__)
Swagger(app)

# 配置日志文件
# log_config()
image_ext = ["bmp", "jpeg", "jpg", "jpe", "png", "tiff"]

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

def predict_img(dataloader,return_info):
    return_info["code"] = 501
    start_time = time.time()
    model = torchvision.models.resnet18(pretrained=True)
    model.eval()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs,5)
    print("模型")
    device = torch.device("cpu")
    model.load_state_dict(torch.load("battle_origin.pth", map_location='cpu'))
    model = model.to(device)
    print("准备")
    res = ['F式战机','J式战机','MIG式战机','other','Su式战机']
    for img,label in dataloader:
        output = model(img)
        print(output)
        _,predicted = torch.max(output.data,1)
    return_info["code"] = 200
    return_info["data"]["labels"] = res[predicted[0]]
    return_info["data"]["size"]["w"] = 200
    return_info["data"]["size"]["h"] = 300
    cost_time = float(round(time.time() - start_time, 3))
    return_info["time"] = cost_time
    return return_info

@app.route('/analysis', methods=['POST'])
@swag_from('swagger_setting.yml')
def detect():  # 接收相关的传输信息
    image = request.files.get('image')
    # 接收json相关信息
    content = request.get_data()

    receive_info = {"code": 500, "msg": "成功", "time": 0,
                   "data": {"size": {"w": 0, "h": 0},
                            "labels": []}}
    flag = 1
    if image:
        file_type = image.content_type.split("/")
        if file_type[0] == 'image' and file_type[1] in image_ext:
            try:
                image = image.read()
                img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
                img_data = Image.fromarray(cv2.cvtColor(
                    img, cv2.COLOR_BGR2RGB))
                img_data.save("./static/01.jpg")
                
            except:
                flag, msg = 0, "上传的图像文件, 读取失败, 请确认图像文件"
        else:
            flag, msg = 0, "上传的文件不是图像文件, 请重新上传图片文件"
    else:
        flag, msg = 0, "参数为空, 请确认输入参数是否缺失"


    if flag:
        f = open('img_label.txt','w')
        path = "./static/01.jpg"
        f.write(str(path)+"  "+str(0)+"\n")
        f.close()
        img_data = MyDataset("img_label.txt")
        data_loader = DataLoader(img_data,batch_size=1,shuffle=True,num_workers=0)
        receive_info = predict_img(data_loader, receive_info)
        logging.info(receive_info)
    else:
        receive_info["msg"] = msg
        logging.error(receive_info)

    return receive_info


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
