from collections import deque
import os

from numpy.core.fromnumeric import clip
import cv2
import math
import numpy as np
from features import get_coordinate
from fms import get_radio
from features import get_label
import simplejson
from dis_and_ang import get_math_information
p=get_math_information()
base_path='data_proc/jump_coordinates/10.txt'
input_path='data_test/coordinates1/'
p3=os.listdir(input_path)

window_size=368*368
MAXLEN=0
NECK=0
L_THIGH=8
R_THIGH=11
L_KNEE=9
R_KNEE=12
L_ANKLE=10
R_ANKLE=13


g=get_coordinate(input_path)
length=len(g)
print(g[0])
con=[[]for i in range(length)]
i=0
for filename in os.listdir(input_path):
    filename=input_path+filename
    with open(filename,'r') as f:
        content=simplejson.load(f)
        
        m=content[5:23]
        
        p1=p.get_skeleton_disiance(m)
        con[i].append(p1)
        i+=1

jump_standard=get_radio(base_path)
JUMP_MODEL=[]
for i in range(len(jump_standard)):
    for j in range(2):
        JUMP_MODEL.append(jump_standard[i][j])
x_deque=deque()
angles=deque()
len=deque()

#print(JUMP_MODEL)

def reset():#两头都可以操作的队列
    x_deque=deque()
    angles=deque()
    len=deque()



def show_joint(x,inx):#得到对应身体的坐标元组
    return x[2*inx],x[2*inx+1]

def get_dis(x1,y1,x2,y2):#得到两点之间的距离
    return ((x1-x2)**2+(y1-y2)**2)**0.5

def get_height(x):#得到脖子到重心的高度
    x0,y0=show_joint(x,NECK)
    x1,y1=show_joint(x,L_THIGH)
    x2,y2=show_joint(x,R_THIGH)
    x3,y3=((x1+x2)/2,(y1+y2)/2)
    return get_dis(x0,y0,x3,y3)

def get_skeleton(x):#得到标准骨架数据
    return x.copy()[2:2+13*2]

def isright(x):#判断数据是否合理
    x0,y0=show_joint(x,NECK)
    x1,y1=show_joint(x,L_THIGH)
    x2,y2=show_joint(x,R_THIGH)
    if x0== None and y0==None:
        return False
    elif x1 ==None and x2==None and x2==None and y2==None:
        return False
    else:
        return True

def isdeque():
    if len(x_deque)>window_size:
        x_deque.popleft()

            
def get_res(x):#得到res
    res=x.copy()
    px=x[0::2]#所有x的坐标
    py=x[1::2]#所有y的坐标
    x0,y0=show_joint(x,NECK)
    height=get_height(x)
    is_lack_knee= show_joint(x,L_KNEE) or show_joint(x,R_KNEE)
    is_lack_ankle=show_joint(x,L_ANKLE) or show_joint(x,R_ANKLE)

    if is_lack_ankle or is_lack_knee:
        for i in range(26):
            if res[i]==np.NAN:
                res[i]=(x0 if i%2==0 else y0)+\
                    height *    JUMP_MODEL[i]

    return res

def get_norm_list(x):
    x0,y0=show_joint(x,NECK)
    x[0::2]=x[0::2]-x0
    x[1::2]=x[1::2]-y0
    return x

def get_pose_1(deque_data,length):
    features=[]
    for i in range(length):
        next_feature=deque_data[i].tolist()
        features+=next_feature
    features=np.array(features)
    return features

def get_center(x_deque,step):#注意力机制
    vel=[]
    if length1>step:
        for i in range(0,length1-step,step):
            dxdy=x_deque[i+step][0:3]-x_deque[i][0:3]
            vel+=dxdy.tolist()
    else:
        vel+=x_deque[0][0:3].tolist()
    return np.array(vel)

def get_all_joint(x_deque,step):
    vel=[]
    if length1>step:
        for i in range(0,length1,step):
            dxdy=x_deque[i+step][:17]-x_deque[i][:17]
            vel+=dxdy.tolist()
    else:
        vel+=x_deque[0][:].tolist()
    return np.array(vel)


'''height_list=[get_height(xi) for xi in x_deque]
#print("height_list:\n",height_list)
mean_height=np.mean(height_list)
#print("mean_height:\n",mean_height)
norm_list=[get_norm_list(xi)/mean_height
           for xi in x_deque]
#print("norm_list:\n",norm_list)
f_pose=get_pose_1(norm_list)
f_v_center=get_center(x_deque,step=1)/mean_height
print(f_v_center)
f_v_center=np.repeat(f_v_center,10)
f_v_joints=get_all_joint(norm_list,step=1)
features=np.concatenate((f_pose,f_v_center,f_v_joints))
print("features:\n",features)'''
length1=0
def get_features(x):#获取特征
    m=p.get_skeleton_disiance(x)
    i=0
    x=get_skeleton(x)
    length1=0
    features=[]
    if isright(x)==False:
        reset()
        return False
    else :
        x=get_res(x)
        x=np.array(x)
        x_deque.append(x)
        length1+=1
        
        height_list=[get_height(xi) for xi in x_deque]
        #print("height_list:\n",height_list)
        mean_height=np.mean(height_list)
        #print("mean_height:\n",mean_height)
        norm_list=[get_norm_list(xi)/mean_height
        for xi in x_deque]
        #print("norm_list:\n",norm_list)
        
        f_pose=get_pose_1(norm_list,length1)
        #print("f_pose:\n",f_pose)
        f_v_center=get_center(x_deque,step=2)/mean_height
        #print(f_v_center)
        f_v_center=np.repeat(f_v_center,16)
        f_v_joints=get_all_joint(norm_list,step=2)
        
        features=np.concatenate((f_pose,f_v_center,f_v_joints,con[i][0]))
        i+=1
        #print("features:\n",features)
        #print("ok")
    return features


def extract_features(input_path):
    x_new=[]
    y_new=[]
    label,clip=get_label(input_path)
    y_new.append(label)
    for i in range(length):
        features=get_features(g[i])
        x_new.append(features)
    x_new=np.array(x_new)
    y_new=np.array(y_new)
    #print(x_new)
    return x_new,y_new

def save_listlist(filepath, ll):
    ''' Save a list of lists to file '''
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(ll, f)

save_path_1='data_test/features_x.csv'
save_path_2='data_test/features_y.csv'
X,Y=extract_features(input_path)
#print("X:\n",X)
#print("Y:\n",Y)
with open(save_path_2,'w') as f:
    for i in range(length):
        f.write(str(Y[0][i]))
        f.write('\n')
    f.close()
np.savetxt(save_path_1,X,fmt="%.5f")
#np.savetxt(save_path_2,Y,fmt="%i")
#save_listlist(save_path_1,X)
    

    

