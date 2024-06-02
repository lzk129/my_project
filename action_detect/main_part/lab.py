import os
import cv2
video_path='data_test/2/seq3.avi'
vc=cv2.VideoCapture(video_path)
save_path='data_test/116/'
os.makedirs(save_path)
start=0

while True:
    rval,frame=vc.read()
    cv2.imwrite(save_path+str(start)+".jpg",frame)
    start+=1

