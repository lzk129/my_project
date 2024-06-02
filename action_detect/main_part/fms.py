import simplejson
import os
import numpy as np
#from first import save_listlist

def read_listlist(filepath):
    ''' Read a list of lists from file '''
    with open(filepath, 'r') as f:
        ll = simplejson.load(f)
        return ll


def save_listlist(filepath, ll):
    ''' Save a list of lists to file '''
    folder_path = os.path.dirname(filepath)
    os.makedirs(folder_path, exist_ok=True)
    with open(filepath, 'w') as f:
        simplejson.dump(ll, f)

base_path='data_proc/jump_coordinates'
save_path='data_proc/jump_skeletons.txt'
nor_path='data_proc/jump_nor.txt'
all_skeletons=[]



def get_radio(base_path):
    p=read_listlist(base_path)
    a=p[5:]
    NECK=1
    L_TIGHT=8
    R_TIGHT=11
    x0=a[NECK][0]
    y0=a[NECK][1]
    y1=a[L_TIGHT][1]
    y2=a[R_TIGHT][1]
    y_m=(y1+y2)/2
    height=abs(y0-y_m)
    for i in range(len(a)):
        for j in range(2):
            a[i][j]/=height
    #print(a)
    return a
    


f=open(save_path,'w')
b=open(nor_path,'w')
for filename in os.listdir(base_path):
    filename= base_path+'/'+filename
    p=read_listlist(filename)
    all_skeletons.append(p)
    g=get_radio(filename)
    f.write(str(p))
    f.write('\n')
    b.write(str(g))
    b.write('\n')
    
    #f.write("\r\n") 
    #f.close()
   
    
    
#save_listlist(save_path,all_skeletons)
#os.makedirs(save_path)
