import numpy as np
from first import do_skeleton
import os
import cv2
from pred import detection

path2 = 'F:/database/'
path1 = 'F:/pri/lzk_project/data/jump/'
def test_skeleton(img_file , res_file):
    do_skeleton(img_file , res_file)

# def detection(path2):
#     dete(path2)

if __name__ == "__main__":
    res_file = path2 + 'test/'
    # os.makedirs(res_file)
    res_file = res_file + '0.jpg'
    path1 = path1 + '00037.jpg'
    # print(path1)
    # img = cv2.imread(path1)
    # print(img)
    # filename
    test_skeleton(path1 , res_file)
    detection(path2)







#data
# # test
