import os
import torch
import numpy as np

from st_gcn import twostream
from joints import normalize_joint,classfication

class TSSTG(object):
    def __init__(self,
                 weight_file='./models/TSSTG/tsstg-model.pt',
                 device='cuda') :
        self.graph_args={'methods':'spatial'}
        self.class_names=['running','falling','jump','fight']
        self.num_class=len(self.class_names)
        self.device=device

        self.model=twostream(self.graph_args,self.num_class).to(self.device)
        self.model.load_state_dict(torch.load(weight_file))
        self.model.eval()

    def predict(self,pts,img_size):
    
        pts[:,:,2]=normalize_joint(pts[:,:,2],img_size[0],img_size[1])
        pts[:,:,2]=classfication(pts[:,:,2])
        pts=np.concatenate((pts,np.expand_dims((pts[:,1,:]+pts[:,2,:])/2,1)),axis=1)
        pts=torch.tensor(pts,dtype=torch.float32)
        pts=pts.permute(2,0,1)[None,:]

        mot=pts[:,:2,1:,:]-pts[:,:2,:-1,:]
        mot=mot.to(self.device)
        pts=pts.to(self.device)
        out=self.model((pts,mot))
        return out.detach().cpu().numpy()
    
