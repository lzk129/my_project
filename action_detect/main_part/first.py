'''
输入：
原始视频帧
输出：
1.骨骼图
2.骨骼坐标
'''
import os
import sys
from matplotlib.pyplot import bone
sys.path.append(r'D:\新建文件夹\lzk_project')
#from tools import get_skeleton
from bone import get_skeleton
from dis_and_ang import get_math_information
import sys
import simplejson
import time
import cv2
import numpy as np
import simplejson
from dis_and_ang import get_math_information
class_name=["handclapping","handwaving","playing"]
tag=2
i=0
flag=0
k=get_math_information()

def do_skeleton(img_file,filename1):
    t = time.time()
        # Fix the input Height and get the width according to the Aspect Ratio
    inHeight = 368
    # print(img_file)
    skeleton=get_skeleton(img_file)
    image1=cv2.imread(img_file)
    frameWidth=image1.shape[1]
    frameHeight=image1.shape[0]
    inWidth = int((inHeight/frameHeight)*frameWidth)

    output=skeleton.generate(image1)
    print("Time Taken in forward pass = {}".format(time.time() - t))

    detected_keypoints = []
    keypoints_list = np.zeros((0,3))
    keypoint_id = 0
    threshold = 0.1
    num=0
    for part in range(skeleton.nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = skeleton.getKeypoints(probMap, threshold)
        #print("Keypoints - {} : {}".format(skeleton.keypointsMapping[part], keypoints))
        num=max(num,len(keypoints))
    
    skeleton_list=[[]for i in range(num)]

    for part in range(skeleton.nPoints):
        probMap = output[0,part,:,:]
        probMap = cv2.resize(probMap, (image1.shape[1], image1.shape[0]))
        keypoints = skeleton.getKeypoints(probMap, threshold)
        #print("Keypoints - {} : {}".format(skeleton.keypointsMapping[part], keypoints))
        num=max(num,len(keypoints))
        for index in range(len(keypoints)):
            skeleton_list[index].append(keypoints[index][:2])
        keypoints_with_id = []
        for i in range(len(keypoints)):
            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
            keypoint_id += 1

        detected_keypoints.append(keypoints_with_id)
    

    frameClone = image1.copy()
    img = np.zeros((frameWidth,frameHeight*2,3), np.uint8)
    # 使用白色填充图片区域,默认为黑色
    img.fill(255)
    for i in range(skeleton.nPoints):
        for j in range(len(detected_keypoints[i])):
            cv2.circle(frameClone, detected_keypoints[i][j][0:2], 5, skeleton.colors[i], -1, cv2.LINE_AA)
    # cv2.imshow("Keypoints",frameClone)

    valid_pairs, invalid_pairs = skeleton.getValidPairs(output,detected_keypoints)
    personwiseKeypoints = skeleton.getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)

    for i in range(17):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(skeleton.POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(img, (B[0], A[0]), (B[1], A[1]), skeleton.colors[i], 3, cv2.LINE_AA)
    
    
    #print(skeleton_list)
    # cv2.putText(img,
    #             f" {num} people in it",(300,150),cv2.FONT_HERSHEY_SIMPLEX,1,(248,248, 255),4, lineType=cv2.LINE_AA)
    # cv2.imshow("Detected Pose" , frameClone)
    cv2.imwrite(filename1,img)
    # cv2.waitKey(10000)
    cv2.destroyAllWindows()
    return skeleton_list,num

def save_listlist(filepath, ll):
    ''' Save a list of lists to file '''
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(ll, f)


def swap(p,len):
    m=p[len]
    
    for i in range(len,0,-1):
        #print("g[i]:\n",g[i])
        #print("i:\n",i)
        p[i]=p[i-1]
    p[0]=m

def get_angle(skeleton_list,num,flag):
    N=8
    angle=[[] for i in range(num)]
    #print(angle)
    do_angle=get_math_information()
    filepath="data_proc/jump_angle/"
    print("flag",flag)
    for i in range(num):
        angle[i]=do_angle.get_skeleton_angle(skeleton_list[i])
        #print(angle[i])
        filepath=filepath+str(flag)+".txt"
        save_listlist(filepath,angle[i])

if __name__ == '__main__':
    path="data_test/116"
    path2="data_test/8"
    print("Program Start!!!")
    tap=862
    a=[]
    a.append(0)
    a.append(1)
    a.append(2)
    a.append(3)
    a.append('c')
    swap(a,4)
    #print(a)
    for filename in os.listdir(path):
        p=[]
        
        
        tap+=1
        
        img_file=path+'/'+filename
        
        filename1=path2+'/'+filename
        p,num=do_skeleton(img_file,filename1)
        file1=filename1.split(".")
        filepath="data_test/coordinates1"+'/'+str(tap)
        flag+=1
        #get_angle(p,num,flag)
        for i in range(num):
            filepath=filepath+str(i)+".txt"
            m=k.get_skeleton_disiance(p[i])
            p[i].append(img_file)
            swap(p[i],len(p[i])-1)
            p[i].append(class_name[tag])
            swap(p[i],len(p[i])-1)
            p[i].append(tag)
            swap(p[i],len(p[i])-1)
            p[i].append(i+1)
            #swap(p[i],len(p[i])-1)
            #p[i].append(num)
            swap(p[i],len(p[i])-1)
            p[i].append(flag)
            swap(p[i],len(p[i])-1)
            #swap(p,len(p))
            '''if m!=None:
                for o in range(len(m)):
                    p[i].append(m[o])'''
        
            #p[i].append(img_file)
            print(p[i])
            if len(p[i])>=23:
                print(p[i])
                save_listlist(filepath,p[i])
            
            #print("img_file:\n",img_file)
        
    

    print("Programs end!!!")


