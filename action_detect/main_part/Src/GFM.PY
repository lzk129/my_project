import torch
import torch.nn as nn

class Gated_Fusion(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.gate3 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.gate4 = nn.Sequential(
            nn.Conv2d(in_channels,in_channels, kernel_size = 1 , padding = 0),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv2d(64,in_channels,kernel_size = 1,padding=0)
        self.conv2 = nn.Conv2d(256,in_channels,kernel_size = 1,padding=0)
        self.conv3 = nn.Conv2d(512,in_channels,kernel_size = 1 , padding = 0)
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear')
        
    
    def forward(self,x,y,z,origin):
        x1 = self.gate1(x)
        # print(origin.size)
        x2 = torch.mul(x,x1)
        
        y1 = self.gate2(y)
        y2 = torch.mul(y,y1)

        z1 = self.gate3(z)
        z2 = torch.mul(z,z1)

        x2 = self.conv1(x2)
        y2 = self.conv2(y2)
        z2 = self.conv3(z2)
        z2 = self.upsample(z2)
        # print(x2.shape)
        # print(y2.shape)
        # print(z2.shape)
        res0 = x2 + y2 + z2

        origin1 = self.gate4(origin) #Gl
        wei1 = 1 - origin1
        wei2 = 1 + origin1

        res1 = wei1 * res0
        res2 = wei2 * origin

        fl = res1 + res2
        # print(fl.shape)
        return fl



    

