import torch
import torch.nn as nn

class Gated(nn.Module):
    def __init__(self,in_channels):
        super().__init__()
        self.gate1 = nn.Sequential(
            nn.Conv2d(32,in_channels,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.gate2 = nn.Sequential(
            nn.Conv2d(1,1,kernel_size=1,padding=0),
            nn.Sigmoid()
        )
        self.downsample = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
        
    #x:(4,32,44,44)
    #y:(4,32,44,44)
    #z:(4,32,88,88)
    #origin:(4,32,44,44)
    def forward(self,x,y,z,origin):
        x = self.gate2(x) #(4,1,44,44)
        y = self.gate2(y) #(4,1,44,44)
        z = self.downsample(z) #(4,32,44,44)
        z = self.gate1(z) #(4,1,44,44)

        f = x + y + z #(4,1,44,44)
        origin1 = self.gate2(origin)
        wei1 = 1 - origin1
        wei2 = 1 + origin1
        res1 = wei1 * f
        res2 = wei2 * origin
        fl = res1 + res2
        return fl


        



    

