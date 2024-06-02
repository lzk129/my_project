import numpy as np
import sys
import os
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from collections import deque
import cv2

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from first import do_skeleton
from smt import get_features
class test(object):
    def __init__(self,model_path,action_labels,human_id=0) :
        self.human_id=human_id
        with open(model_path,'rb') as f:
            self.model=pickle.load(f)
        self.action_labels=action_labels
        self.thresh=0.5
        self.reset()
    
    def reset(self):
        self.scores_list=deque()
        self.scores=None


    def predict(self,skeleton):
        label=" "
        features=get_features(skeleton)
        print("features:\n",features)
        features=features.reshape(-1,features.shape[0])
        curr_scores=self.model.predict_prob(features)[0]
        self.scores=self.smooth_scores(curr_scores)
        if self.scores.max()<self.thresh:
            predict_label=label
        else:
            predict_index=self.scores.argmax()
            predict_label=self.action_labels[predict_index]
        return predict_label
    
    def smooth_scores(self,curr_scores):
        self.scores_list.append(curr_scores)
        max=2
        if len(self.scores_list)>max:
            self.scores_list.popleft()
        
        score_sums=np.zeros((len(self.action_labels),))
        for score in self.scores_list:
            score_sums+=score
        score_sums/=len(self.scores_list)
        print("准确率:\n",score_sums)
        return score_sums
    
    def draw(self,img_disp):
        if self.scores is None:
            return
        for i in range(-1,len(self.action_labels)):
            font_size=0.7
            txt_x=20
            txt_y=150+i*30
            color=255
            if i==-1:
                s="P{}:".format(self.human_id)
            else:
                label=self.action_labels[i]
                s="{:<5}:{:.2f}".format(label,self.scores[i])
                color*=(0.0+1.0*self.scores[i])**0.5
            
            cv2.putText(img_disp,text=s,org=(txt_x,txt_y),fontFace=cv2.FONT_HERSHEY_SIMPLEX,fontScale=font_size,
            color=(0,0,int(color)),thickness=2)




        