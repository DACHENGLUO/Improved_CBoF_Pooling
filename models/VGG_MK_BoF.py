import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


class BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(BoF_Pooling, self).__init__()
        
        self.n_codewords = n_codewords
        self.weight = nn.Parameter(torch.rand(n_codewords,in_channels),requires_grad=True)
        self.gamma = nn.Parameter(1e-6 * torch.ones((n_codewords)), requires_grad=True) 
        
        self.norm1 = nn.BatchNorm2d(n_codewords)
        #self.norm2 = nn.LayerNorm(n_codewords, eps=1e-6) # final norm layer
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        x_square = torch.sum(torch.square(x),dim=3,keepdim=True)
        y_square = torch.sum(torch.square(self.weight),dim=1)
        y_square = y_square.view(1,1,1,self.n_codewords)
        
        x = x_square + y_square - 2 * F.linear(x, self.weight)
        #x = torch.square(self.gamma) * x
        x = self.gamma * x
        x = F.softmax(-x,dim=3)
        

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm1(x)
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x
class Group_BoF_Pooling(nn.Module):

    def __init__(self,in_channels: int, n_codewords: int):
        super(Group_BoF_Pooling,self).__init__()
        #self.n_codewords = n_codewords

        self.Pooling1 = BoF_Pooling(in_channels,int(n_codewords/4))
        self.Pooling2 = BoF_Pooling(in_channels,int(n_codewords/2))
        self.Pooling3 = BoF_Pooling(in_channels,n_codewords)
        
    def forward(self, x):
        
        histogram1 = self.Pooling1(x)
        histogram2 = self.Pooling2(x)
        histogram3 = self.Pooling3(x)
       
        return torch.cat((histogram1,histogram2,histogram3),1)

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, n_codewords, num_class=100):
        super().__init__()
        self.features = features
        self.bofpool = Group_BoF_Pooling(512,n_codewords)
        self.fc = nn.Linear(n_codewords+int(n_codewords/2)+int(n_codewords/4), num_class)

        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = self.bofpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

# def vgg11_bn_bof():
#     return VGG(make_layers(cfg['A'], batch_norm=True))

# def vgg13_bn_bof():
#     return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn_mkbof_512():
    return VGG(make_layers(cfg['D'], batch_norm=True), n_codewords=512)
def vgg16_bn_mkbof_256():
    return VGG(make_layers(cfg['D'], batch_norm=True), n_codewords=256)
def vgg16_bn_mkbof_128():
    return VGG(make_layers(cfg['D'], batch_norm=True), n_codewords=128)
def vgg16_bn_mkbof_64():
    return VGG(make_layers(cfg['D'], batch_norm=True), n_codewords=64)

# def vgg19_bn_bof():
#     return VGG(make_layers(cfg['E'], batch_norm=True))



