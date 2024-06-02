import sys
import numpy as np

def normalize_joint(xy,width,height,flip=False):
    if xy.ndim==2:
        xy=np.expand_dims(xy,0)
    xy[:,:,0]/=width
    xy[:,:,1]/=height
    if flip:
        xy[:,:,0]=1-xy[:,:,0]
    return xy

def classfication(xy):
    if xy.ndim==2:
        xy=np.expand_dims(xy,0)
    xy_min=np.nanmin(xy,axis=1)
    xy_max=np.nanmax(xy,axis=1)
    for i in range(xy.shape[0]):
        xy[i]=((xy[i]-xy_min[i])/(xy_max[i]-xy_min[i]))*2-1
    return xy.squeeze()