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

class model(object):
    def __init__(self) :
        self._init_all_models()
        self.clf=self._choose_model("Neural Net")
        self.num_from_frame=40#修改

    def predict(self,x):
        y_predict=self.clf.predict(self.pca.transform(x))
        return y_predict

    def predict_and_evaluate(self,test_x,test_y):
        test_y_predict=self.predict(test_x)
        N=len(test_y)
        n=sum(test_y_predict==test_y)
        accu=n/N
        return accu,test_y_predict

    def train(self,x,y):
        n_components=min(self.num_from_frame,x.shape[1])
        self.pca=PCA(n_components=n_components,whiten=True)
        self.pca.fit(x)
        print("Sum big values:\n",np.sum(self.pca.explained_variance_ratio_))
        x_new=self.pca.transform(x)
        print("After PCA,x.shape=\n",x_new.shape)
        self.clf.fit(x_new,y)

    def _choose_model(self,name):
        self.model_name=name
        index=self.names.index(name)
        return self.classifiers[index]

    def _init_all_models(self):
        self.names=["Nearest Neighbors","Linear SVM","RBF SVM","Gaussian Process",
                    "Decision Tree","Random Forest","Neural Net","AdaBoost","Naive Bayes","RDA"]
        self.model_name=None
        self.classifiers=[
            KNeighborsClassifier(5),
            SVC(kernel="linear",C=10.0),
            SVC(gamma=0.01,C=1.0,verbose=True),
            GaussianProcessClassifier(1.0*RBF(1.0)),
            DecisionTreeClassifier(max_depth=5),
            RandomForestClassifier(max_depth=30,n_estimators=100,max_features="auto"),
            MLPClassifier((20,30,40)),
            AdaBoostClassifier(),
            GaussianNB(),
            QuadraticDiscriminantAnalysis()
        ]

    def predict_prob(self,x):
        y_prob=self.clf.predict_proba(self.pca.transform(x))
        return y_prob
