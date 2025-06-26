import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda:1'

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Upsample(nn.Module):
    """ nn.Upsample is deprecated """

    def __init__(self, scale_factor, mode="bilinear"):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=True, recompute_scale_factor=True)
    
    # Index Pooling
class pool(nn.Module):
  def __init__(self,channels):
    super(pool, self).__init__()
    self.channels=channels
    
    self.weight1 = torch.zeros((channels,1,2,2)).to(device)
    self.weight2 = torch.zeros((channels,1,2,2)).to(device)
    self.weight3 = torch.zeros((channels,1,2,2)).to(device)
    self.weight4 = torch.zeros((channels,1,2,2)).to(device)
    self.weight1[:,:,0,0]=1
    self.weight2[:,:,0,1]=1
    self.weight3[:,:,1,0]=1
    self.weight4[:,:,1,1]=1
  def forward(self, x):
    with torch.no_grad():
         x1=F.conv2d(x, self.weight1,stride=2,groups=self.channels, bias=None)
         x2=F.conv2d(x, self.weight2,stride=2,groups=self.channels, bias=None)
         x3=F.conv2d(x, self.weight3,stride=2,groups=self.channels, bias=None)
         x4=F.conv2d(x, self.weight4,stride=2,groups=self.channels, bias=None)
    return x1,x2,x3,x4
  

#DAMIP Module
class attn_pool(nn.Module):
    def __init__(self,feature_channels):
       super(attn_pool, self).__init__()
       self.pool1=pool(feature_channels)
       self.pool2=pool(1)
       self.conv1 = nn.Conv2d(feature_channels*4, 2*feature_channels, kernel_size=1, stride=1, bias=False)
       self.conv2 = nn.Conv2d(2*feature_channels,2*feature_channels,kernel_size=7,stride=1,padding=3)
       self.bn2 = nn.BatchNorm2d(2*feature_channels)
       self.a1=nn.Parameter(torch.Tensor(1))
       self.a2=nn.Parameter(torch.Tensor(1))
       self.a3=nn.Parameter(torch.Tensor(1))
       self.a4=nn.Parameter(torch.Tensor(1))
    def forward(self,map,feature):
        feature1,feature2,feature3,feature4=self.pool1(feature)
        map1,map2,map3,map4=self.pool2(map)

        fm1 = self.a1*feature1 + feature1*map1
        fm2 = self.a2*feature2 + feature2*map2
        fm3 = self.a3*feature3 + feature3*map3
        fm4 = self.a4*feature4 + feature4*map4

        mat=torch.cat((fm1,fm2,fm3,fm4),1)
        mat=self.conv1(mat)
        mat=F.relu(self.bn2(self.conv2(mat)))
        return mat
    

# DPMG Module
class dsup(nn.Module):
  def __init__(self,input_channels):
    super(dsup,self).__init__()
    self.conv1=conv3x3(input_channels,input_channels//2)
    self.bn1=nn.BatchNorm2d(input_channels//2)
    self.conv2=conv3x3(input_channels//2,32)
    self.conv3=nn.Conv2d(32,1,kernel_size=3,stride=1,padding=1)
  def forward(self,x):
    x=F.relu(self.bn1(self.conv1(x)))
    x=self.conv2(x)
    x=self.conv3(x)
    return torch.sigmoid(x)
  

# Dilated Convolutional Block
class conv_enc(nn.Module):
  def __init__(self,in_channels,out_channels,dil):
    super(conv_enc,self).__init__()
    self.conv1 = conv1x1(in_channels,in_channels)
    self.bn1 = nn.BatchNorm2d(in_channels)
    self.conv2 = conv3x3(in_channels,in_channels,dilation=dil)
    self.bn2 = nn.BatchNorm2d(in_channels)
    self.conv3 = conv1x1(in_channels,out_channels)
    self.bn3 = nn.BatchNorm2d(out_channels)
    self.conv4 = conv3x3(out_channels,out_channels,dilation=dil)
    self.bn4 = nn.BatchNorm2d(out_channels)
    # input dimension matching
    self.conv0 = conv1x1(in_channels,out_channels)
  def forward(self,x):
    identity = self.conv0(x)

    x=F.relu(self.bn1(self.conv1(x)))
    x=F.relu(self.bn2(self.conv2(x)))
    x=F.relu(self.bn3(self.conv3(x)))
    x=self.bn4(self.conv4(x))

    return F.relu(x+identity)
  

class enc(nn.Module):
  def __init__(self,input_channels,output_channels,dil):
    super(enc,self).__init__()
    self.conv1 = conv_enc(input_channels,output_channels,dil)
    self.conv2 = conv_enc(output_channels,output_channels,2*dil)
    self.dp_sup = dsup(output_channels)

  def forward(self,x):
    x1 = self.conv1(x)
    x1 = self.conv2(x1)
    x1_out = self.dp_sup(x1)
    return x1, x1_out
  

# DAMSCA Module
class kqcbam(nn.Module):
  def __init__(self,input_channels,scale_factor=2):
    super(kqcbam,self).__init__()
    self.conv1=nn.Conv2d(1,input_channels,kernel_size=1)
    self.gap=nn.AdaptiveAvgPool2d((1,1))
    self.upsample=Upsample(scale_factor)
  def forward(self,map,feature):
    f1=map*feature
    map2=self.conv1(map)
    map2=self.gap(map2)
    f2=torch.sigmoid(map2)*feature
    out=F.relu(f1+f2)
    return self.upsample(out)
  

# Decoder Module
class decoder(nn.Module):
  def __init__(self,input_channels):
    super(decoder,self).__init__()
    self.conv1=nn.ConvTranspose2d(input_channels,128,kernel_size=4,stride=2,padding=1)
    self.bn1=nn.BatchNorm2d(128)
    self.conv2=nn.Conv2d(128,64,3,stride=1,padding=1)
    self.bn2=nn.BatchNorm2d(64)
    self.conv3=nn.Conv2d(64,32,3,stride=1,padding=1)
    self.conv_out=nn.Conv2d(32,1,3,stride=1,padding=1)
  def forward(self,x):
    x=F.relu(self.bn1(self.conv1(x)))
    x=F.relu(self.bn2(self.conv2(x)))
    x=self.conv3(x)
    x=self.conv_out(x)
    return torch.sigmoid(x)
  

# MSSDMPA-Net
class dsmpnet(nn.Module):
  def __init__(self,input_channels):
    super(dsmpnet,self).__init__()
    self.conv1=nn.Conv2d(input_channels,64,kernel_size=7,stride=2,padding=3)
    self.bn1=nn.BatchNorm2d(64)

    self.pool1=attn_pool(64)
    self.pool2=attn_pool(128)
    self.pool3=attn_pool(256)

    self.path1=enc(64,64,1)
    self.path2=enc(128,128,2)
    self.path3=enc(256,256,3)
    self.path4=enc(512,512,4)

    self.cbm1=kqcbam(64,1)
    self.cbm2=kqcbam(128,2)
    self.cbm3=kqcbam(256,4)
    self.cbm4=kqcbam(512,8)

    self.decoder=decoder(960)

  def forward(self,x):
    x=F.relu(self.bn1(self.conv1(x)))
    x1,x1_out=self.path1(x)
    x=self.pool1(x1_out,x)
    x2,x2_out=self.path2(x)
    x=self.pool2(x2_out,x)
    x3,x3_out=self.path3(x)
    x=self.pool3(x3_out,x)
    x4,x4_out=self.path4(x)

    x1=self.cbm1(x1_out,x1)
    x2=self.cbm2(x2_out,x2)
    x3=self.cbm3(x3_out,x3)
    x4=self.cbm4(x4_out,x4)
    x_out=torch.cat((x1,x2,x3,x4),1)
    x_out=self.decoder(x_out)
    return x_out,x1_out,x2_out,x3_out,x4_out
