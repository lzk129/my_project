import numpy as np
import cv2
import argparse
import pickle
from net import model
import os
from first import do_skeleton
from smt import extract_features
from smt import get_features
from train import test
input_path='data_test/lzk.pickle'
f=open(input_path,'rb') 
s=f.read()
#models=model()
classes=["handclapping","handwaving"]
features_path='data_proc/jump_x.csv'
label_path='data_proc/jump_y.csv'
save_path='data_test/1/'
save_path_2='data_test/2/'
capture=cv2.VideoCapture(0)
begin=0
test=test(input_path,classes)
#cv2.namedWindow("Image",0)
#cv2.resizeWindow("Image",600,600)
'''while(True):
    #ret,frame=capture.read()
    filepath=save_path+str(begin)+".jpg"
    filepath1=save_path_2+str(begin)+".jpg"
    begin+=1
    #cv2.imwrite(filepath,frame)
    p,num=do_skeleton(filepath,filepath1)
    print("坐标是:\n",p)
    print(f"图中有{num}个人")
    coordinate=[]
    for i in range(len(p)):
        if len(p[i])>=0:
            for j in range(len(p[i])):
                for k in range(2):
                    coordinate.append(p[i][j][k])
    a=get_features(coordinate)
    print(a)           
    #cv2.imshow("Image",frame)
    #cv2.waitKey(1000)'''
    


'''x=np.loadtxt(features_path,dtype=float)
y=np.loadtxt(label_path,dtype=int)
print(models.predict(x))
g=models.predict(x)
classes=['run','jump','sit','stand']
#for i in range(len(g)):
    #print("这一帧显示的动作是:\n",classes[g[i]])
do_path='data/run'
cv2.namedWindow("Image",0)
cv2.resizeWindow("Image",600,600)
start=0'''
do_path='data_test/6'
cv2.namedWindow("Image",0)
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
#cv2.resizeWindow("Image",3000,3000)
sum=0
sum1=0
for filename in os.listdir(do_path):
    if filename.endswith(".jpg"):
        sum+=1
        filename=do_path+'/'+filename
        filename1=save_path_2+filename
        a=cv2.imread(filename)
        p,num=do_skeleton(filename,filename1)
        coordinate=[]
        for i in range(len(p)):
            if len(p[i])>=18:
                for j in range(18):
                    for k in range(2):
                        coordinate.append(p[i][j][k])
        #print(coordinate)
        
        #print(features)
        #features=features[None,:]
        c=test.predict(coordinate)
        img1=cv2.imread(filename)
        test.draw(img1)
        print(c)
        if c=='handwaving':
            sum1+=1

        font=cv2.FONT_HERSHEY_SIMPLEX
        img=cv2.putText(a,f"The action is {c}",(50,100),font,0.2,(255,0,255),1)
        #start+=1
        cv2.imshow("Image",a)
        cv2.waitKey(1000)

print("本次测试的准确率是:\n",sum1/sum)


