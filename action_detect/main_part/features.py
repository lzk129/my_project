import os
#from posix import listdir
import sys
from numpy.lib import RankWarning
import simplejson
import json

def save(A,B):
    with open(A,'w') as f:
        simplejson.dump(B)



def get_label(filepath):
    g=[]
    m=[]
   
    for filename in os.listdir(filepath):
        if filename.endswith('.txt'):
            filename=filepath+'/'+filename 
            #print(filename)
            with open(filename,'r') as f:
                p=simplejson.load(f)
                g.append(p[2])
                m.append(p[1])
           
    return g,m

def get_coordinate(filepath):
    file_path=os.listdir(filepath)
    length=len(file_path)
    g=[[] for i in range(length)]
    l=0
    for filename in os.listdir(filepath):
        if filename.endswith('.txt'):
            filename=filepath+'/'+filename
            with open(filename,'r') as f:
                p=simplejson.load(f)
                o=p[5:]
                
                for i in range(18):
                    for j in range(2):
                        g[l].append(o[i][j])
                
            l+=1
    #print("zhejiushi:\n",g)
    return g




           
            
