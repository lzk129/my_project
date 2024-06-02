
from numpy.core.fromnumeric import size
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
sys.path.append(r'D:\新建文件夹\lzk_project\main_part')
from second import Graph
from third import tgcn

class st_gcn(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True) :
        super().__init__()

        #assert(len(kernel_size)==2)
        #assert (kernel_size[0]%2==1)
        padding=((kernel_size[0]-1)//2,0)
        self.gcn=tgcn(in_channels,out_channels,kernel_size[1])

        self.tcn=nn.Sequential(
            nn.BatchNorm2d(out_channels,affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,
                    out_channels,
                    (kernel_size[0],1),
                    (stride,1),
                    padding),
            nn.BatchNorm2d(out_channels),
            #nn.Conv2d(out_channels,out_channels,(kernel_size[0],1),(stride,1),padding),
            #nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout,inplace=True),
            
        )

        if not residual:
            self.residual=lambda x:0
        elif (in_channels==out_channels) and (stride==1):
            self.residual=lambda x:x
        else:
            self.residual=nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride,1)
                ),
                nn.BatchNorm2d(out_channels),

            )
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self,x,A):
        res=self.residual(x)
        self.res=res
        x,A=self.gcn(x,A)
        x=self.tcn(x)+res
        return self.relu(x),A
class models(nn.Module):
    def __init__(self,
                 in_channels=2,
                 num_class=3,
                 method='spatial',
                 edge_importance_weighting=True,
                 **kwargs):
        super().__init__()

        self.graph=Graph(methods=method)
        A=torch.tensor(self.graph.A,dtype=torch.float32,requires_grad=False)
        self.register_buffer('A',A)

        spatial_kernel_size=A.size(0)
        temporal_kernel_size=9
        kernel_size=(temporal_kernel_size,spatial_kernel_size)
        self.data_bn=nn.BatchNorm1d(in_channels*18)
        #kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}#新增加的
        self.st_gcn_net=nn.ModuleList((
            st_gcn(in_channels,64,kernel_size,1,residual=False,**kwargs),
            st_gcn(64,64,kernel_size,1,**kwargs),
            st_gcn(64,64,kernel_size,1,**kwargs),
            st_gcn(64,64,kernel_size,1,**kwargs),
            st_gcn(64,128,kernel_size,2,**kwargs),
            st_gcn(128,128,kernel_size,1,**kwargs),
            st_gcn(128,128,kernel_size,1,**kwargs),
            st_gcn(128,256,kernel_size,2,**kwargs),
            st_gcn(256,256,kernel_size,1,**kwargs),
            st_gcn(256,256,kernel_size,1,**kwargs),
        ))
        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.st_gcn_net
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_net)
        self.fcn=nn.Conv2d(256,num_class,kernel_size=1)
    
    def forward(self,x):
        N,C,T,V,M=x.size()
        x=x.permute(0,4,3,1,2).contiguous()
        x=x.view(N*M,V*C,T)
        x=self.data_bn(x)
        x=x.view(N,M,V,C,T)
        x=x.permute(0,1,3,4,2).contiguous()
        x=x.view(N*M,C,T,V)

        for gcn,importance in zip(self.st_gcn_net,self.edge_importance):
            x,_=gcn(x,self.A*importance)

        x=F.avg_pool2d(x,x.size()[2:])
        x=x.view(N,M,-1,1,1).mean(dim=1)

        x=self.fcn(x)
        x=x.view(x.size(0),-1)

        return x

    def extract_feature(self,x):
        N,C,T,V,M=x.size()
        x=x.permute(0,4,3,1,2).contiguous()
        x=x.view(N*M,V*C,T)
        x=self.data_bn(x)
        x=x.view(N,M,V,C,T)
        x=x.permute(0,1,3,4,2).contiguous()
        x=x.view(N*M,C,T,V)

        for gcn,importance in zip(self.st_gcn_net,self.edge_importance):
            x,_=gcn(x,self.A*importance)
        
        _,c,t,v=x.size()
        feature=x.view(N,M,c,t,v).permute(0,2,3,4,1)

        x=self.fcn(x)
        output=x.view(N,M,-1,t,v).permute(0,2,3,4,1)

        return output,feature
    
class twostream(nn.Module):
    def __init__(self,
                 graph_args,
                 num_class,
                 edge_importance_weighting=True,
                 **kwargs
                 ) :
        super().__init__()
        self.pts_stream=twostream(3,graph_args,None,edge_importance_weighting,**kwargs)
        self.mot_stream=twostream(2,graph_args,None,edge_importance_weighting,**kwargs)

        self.pcn=nn.Linear(256*2,num_class)
       
    def forward(self,inputs):
        out1=self.pts_stream(inputs[0])
        out2=self.mot_stream(inputs[1])
        concat=torch.cat([out1,out2],dim=-1)
        out=self.pcn(concat)
        return torch.sigmoid(out)


             

