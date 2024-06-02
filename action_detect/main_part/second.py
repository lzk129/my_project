import math
import numpy as np
import sys
#sys.path.append(r'D:\新建文件夹\lzk_project')
from bone import get_skeleton
from dis_and_ang import get_math_information
import os
import networkx as nx
path="data/run/00041.jpg"
ske=get_skeleton(path)
class Graph():
    def __init__(self,hop=1,t=1,methods='uniform') :
        self.link=[(4, 3), (3, 2), (7, 6), (6, 5), (13, 12), (12,11),
(10, 9), (9, 8), (11, 5), (8, 2), (5, 1), (2, 1),
(0, 1), (15, 0), (14, 0), (17, 15), (16, 14)]
        self.edge=[(i,i)for i in range(ske.nPoints,ske.nPoints)]+self.link
        self.num=ske.nPoints
        self.hop=hop
        self.matrix=get_hop_dis(self.num,self.edge,self.hop)
        self.t=t
        self.center=1
        self.get_adjacency(methods)

        
    def get_normalize_1(self,matrix):
        a=np.sum(matrix,0)
        D=np.zeros((ske.nPoints,ske.nPoints))
        for i in range(ske.nPoints):
            if a[i]>0:
                D[i,i]=a[i]**(-1)

        X=np.dot(matrix,D)
        return X

    def get_normalize_0(self,matrix):
        a=np.sum(matrix,0)
        D=np.zeros((ske.nPoints,ske.nPoints))
        for i in range(ske.nPoints):
            if a[i]>0:
                D[i,i]=a[i]**(-0.5)
        X=np.dot(np.dot(D,matrix),D)
        return X

    def get_adjacency(self,methods):
        tuple0=range(0,self.hop+1,self.t)
        adjacency=np.zeros((self.num,self.num))
        for hop in tuple0:
            adjacency[self.matrix==hop]=1
        normalize_matrix=self.get_normalize_1(adjacency)

        if methods=='uniform':
            A=np.zeros((1,self.num,self.num))
            A=normalize_matrix
            self.A=A
        elif methods=='distance':
            A=np.zeros((len(tuple0),self.num,self.num))
            for i,j in enumerate(tuple0):
                A[i][self.matrix==j]=normalize_matrix[self.matrix==j]
            self.A=A
        elif methods=="spatial":
            A=[]
            for hop in tuple0:
                a_root=np.zeros((self.num,self.num))
                a_close=np.zeros((self.num,self.num))
                a_further=np.zeros((self.num,self.num))
                self.root=a_root
                self.close=a_close
                self.further=a_further
                for i in range(self.num):
                    for j in range(self.num):
                        if self.matrix[j,i]==hop:
                            if self.matrix[j,self.center]==self.matrix[i,self.center]:
                                a_root[j,i]=normalize_matrix[j,i]
                            elif self.matrix[j,self.center]>self.matrix[i,self.center]:
                                a_close[j,i]=normalize_matrix[j,i]
                            elif self.matrix[j,self.center]<self.matrix[i,self.center]:
                                a_further[j,i]=normalize_matrix[j,i]
                if hop==0:
                    A.append(a_root)
                else:
                    A.append(a_root+a_close)
                    A.append(a_further)
            A=np.stack(A)
            self.A=A
        else:
            raise ValueError("Do Not Exist This Strategy")

def get_hop_dis(num,edge,hop):
    a=np.zeros((num,num))
    for i,j in edge:
        a[i][j]=a[j][i]=1
    b=np.zeros((num,num))+np.inf
    mat=[np.linalg.matrix_power(a,d)for d in range(2)]
    mat1=(np.stack(mat)>0)
    for d in range(1,-1,-1):
        b[mat1[d]]=d
    return b






