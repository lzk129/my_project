from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from torchvision import transforms
from PIL import Image
from show import test_skeleton
from pred import detection
from flask import make_response
import cv2
import numpy as np

import os
app = Flask(__name__)


path2 = 'F:/database/'
path1 = 'F:/pri/lzk_project/data/jump/0.jpg'

@app.route('/show', methods=['POST'])
def show():
    a = request.files.get('file')
    img = a
    print(img)
    cv2.imwrite(path1 , img)
    path = path2 + 'test' 
    os.removedirs(path)
    os.makedirs(path)
    path = path + '/0.jpg'
    test_skeleton(path1 , path)
    return path


@app.route('/predict', methods=['POST'])
def predict():
    # 获取前端传输的图片
    file = request.files['file']
    

    # 对图片进行预处理
    label = detection(path2)

    # 返回预测结果
    return jsonify({'result': label})


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8088)