import shutil
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt

import sys
from PyQt5.QtWidgets import *
from PyQt5 import QtGui
import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
import os
from pred import detection
from first import do_skeleton

path2 = 'F:/database/'
path1 = 'F:/pri/lzk_project/data/jump/0.jpg'
class picture(QWidget):
    def __init__(self):
        super(picture, self).__init__()
        # self.setStyleSheet("background-color:rgb(100,90,60)")
        self.font = QFont()
        self.font.setFamily("SimHei")
        self.font.setBold(1)  # 设置为粗体
        self.font.setPixelSize(24)
        self.resize(1600, 900)
        self.setWindowTitle("人体行为识别检测平台")

        self.label1 = QLabel(self)
        self.label1.setText("   待检测图片")
        self.label1.setFixedSize(700, 500)
        self.label1.move(110, 80)

        self.label2 = QLabel(self)
        self.label2.setText("   骨骼检测结果")
        self.label2.setFixedSize(700, 500)
        self.label2.move(850, 80)

        main_layout = QHBoxLayout()
        main_layout.addWidget(self.label1,1, Qt.AlignVCenter)
        main_layout.addWidget(self.label2,1, Qt.AlignVCenter)
        main_layout.setSpacing(40)

        self.label3 = QLabel(self)
        self.label3.setText("识别结果")
        self.label3.setStyleSheet("font-size:20px;")
        self.label3.move(750,650)
        self.label3.setFont(self.font)
        self.label3.adjustSize()

        btn = QPushButton(self)
        btn.setText("打开图片")
        btn.move(10, 20)
        btn.clicked.connect(self.open_image)

        btn1 = QPushButton(self)
        btn1.setText("检测图片")
        btn1.move(170, 20)
        btn1.clicked.connect(self.predict)

        btn3 = QPushButton(self)
        btn3.setText("评估指标")
        btn3.move(330, 20)
        btn3.clicked.connect(self.evaluate)

    def open_image(self):
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        '''上面一行代码是弹出选择文件的对话框，第一个参数固定，第二个参数是打开后右上角显示的内容
            第三个参数是对话框显示时默认打开的目录，"." 代表程序运行目录
            第四个参数是限制可打开的文件类型。
            返回参数 imgName为G:/xxxx/xxx.jpg,imgType为*.jpg。	
            此时相当于获取到了文件地址 
        '''
        # print(imgName.find('Imgs'))
        # print(imgName[38])
        # os.removedirs(path2)
        shutil.rmtree(path2)
        os.makedirs(path2 + 'test/')
        path = path2 + 'test/0.jpg'
        imgName_cv2=cv2.imread(imgName)
        # gt = cv2.imread()
        im0= cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        do_skeleton(imgName , path)
        # img1 = cv2.imread(imgName)
        # # print(img1)
        # cv2.imwrite(save_path1+'1.png',img1)
        #这里使用cv2把这张图片读取进来，也可以用QtGui.QPixmap方式。然后由于cv2读取的跟等下显示的RGB顺序不一样，所以需要转换一下顺序
        showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        self.label1.setPixmap(QtGui.QPixmap.fromImage(showImage))
        # shutil.rmtree(save_path)

    def predict(self):
        imgName = path2 + 'test/0.jpg'
        imgName_cv2=cv2.imread(imgName)
        # gt = cv2.imread()
        im0= cv2.cvtColor(imgName_cv2, cv2.COLOR_RGB2BGR)
        showImage = QtGui.QImage(im0, im0.shape[1], im0.shape[0], 3 * im0.shape[1], QtGui.QImage.Format_RGB888)
        self.label2.setPixmap(QtGui.QPixmap.fromImage(showImage))
        label = detection(path2)
        self.label = label

    def evaluate(self):
        self.label3.setFont(self.font)
        self.label3.setText("label:{} ".format(str(self.label)))
        self.label3.setFixedHeight(50)
        self.label3.setFixedWidth(120)
        self.label3.setParent(self)

        




if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui_p = picture()
    ui_p.show()
    sys.exit(app.exec_())