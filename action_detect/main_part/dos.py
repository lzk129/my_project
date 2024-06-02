import cv2
import os
import numpy as np
import sys

fc=cv2.VideoCapture(0)
while True:
    val,frame=fc.read()
    cv2.imshow("Image",frame)
    cv2.waitKey(1000)
