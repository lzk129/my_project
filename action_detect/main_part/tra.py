import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from numpy.lib import loadtxt
import sklearn.model_selection
from sklearn.metrics import classification_report
import cv2
from net import model
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
features_path='data_test/features_x.csv'
label_path='data_test/features_y.csv'




def train_test_spilt(x,y,radio):
    is_spilt=True
    if is_spilt:
        rand_speed=1
        train_x,test_x,train_y,test_y=sklearn.model_selection.train_test_split(x,y,test_size=radio,random_state=rand_speed)
    else:
        train_x=np.copy(x)
        test_x=np.copy(x)
        train_y=y.copy()
        test_y=y.copy()
    return train_x,train_y,test_x,test_y

def plot_confusion_matrix(y_true,y_predict,classes,normalize=False,title=None,cmap=plt.cm.Blues,size=None):
    if not title:
        if normalize:
            title='Normalize confusion matrix'
        else:
            title='Confusion matrix,without normalization'
    cm=confusion_matrix(y_true,y_predict)
    classes=classes
    if normalize:
        cm=cm.astype('float')/cm.sum(axis=1)[:,np.newaxis]
        print("Display normalized confusion matrix ...")
    else:
        print("Display confusion matrix without normalization ...")

    fig,ax=plt.subplots()
    if size is None:
        size=(12,8)
    fig.set_size_inches(size[0],size[1])
    im=ax.imshow(cm,interpolation='nearest',cmap=cmap)
    ax.figure.colorbar(im,ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
    yticks=np.arange(cm.shape[0]),
    xticklabels=classes,
    yticklabels=classes,
    ylabel='True label',
    xlabel='Predicted label')
    ax.set_ylim([-0.5,len(classes)-0.5])
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right',rotation_mode='anchor')
    fmt='.2f' if normalize else 'd'
    thresh=cm.max()/2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i,j],fmt),
                    ha='center',va='center',
                    color='white' if cm[i,j]>thresh else 'black')
    fig.tight_layout()
    return ax,cm


def evaluate_model(model,classes,train_x,train_y,test_x,test_y):
    t0=time.time()
    train_accuracy,train_y_predict=model.predict_and_evaluate(train_x,train_y)
    test_accuracy,test_y_predict=model.predict_and_evaluate(test_x,test_y)
    print(f'Accuracy on trainging set is {train_accuracy}')
    print(f"Accuracy on testing set is{test_accuracy}")
    print("Accuracy report:")
    print(classification_report(test_y,test_y_predict,target_names=classes,output_dict=False))
    average_time=(time.time()-t0)/(len(train_y)+len(test_y))
    print("Time cost for predicting one sample:"
          "{:.5f} seconds".format(average_time))
    
    axis,cf=plot_confusion_matrix(test_y,test_y_predict,classes,normalize=False,size=(12,8))
    plt.show()

print("\nReading csv files of classes,features,and labels ...")
x=np.loadtxt(features_path,dtype=float)
y=np.loadtxt(label_path,dtype=int)
#print(x)
train_x,train_y,test_x,test_y=train_test_spilt(x,y,radio=0.3)
print("训练的格式是：\n",train_x.shape)
print("训练的数量是:\n",len(train_y))
print("测试的数量是:\n",len(test_y))
print("现在开始训练:\n")
model=model()
model.train(train_x,train_y)
print("开始评估模型:\n")
classes=["handclapping","handwaving","playing"]
evaluate_model(model,classes,train_x,train_y,test_x,test_y)
save_path='data_test/lzk.pickle'
with open(save_path,'wb') as f:
    pickle.dump(model,f)






