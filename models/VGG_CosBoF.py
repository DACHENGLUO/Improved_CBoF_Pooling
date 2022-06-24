import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-8)
    return x
# def normalize(x, axis=-1):
#     x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-8)
#     return x


# class Cos_BoF_Pooling(nn.Module):
#     def __init__(self,in_channels: int, n_codewords: int):
#         super(Cos_BoF_Pooling, self).__init__()
#         self.n_codewords = n_codewords
#         self.weight = nn.Parameter(torch.rand(n_codewords,in_channels,1,1),requires_grad=True)
#         self.norm1 = nn.BatchNorm2d(n_codewords)
#         #self.sigmas = nn.Parameter(torch.full((1, n_codewords, 1, 1), 0.1),requires_grad=True)
#         nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

#     def forward(self, x):
#         x_norm = normalize(x,axis=1)
#         y_norm = normalize(self.weight,axis=1)

#         #dists = K.maximum(dists, 0)

#         # Quantize the feature vectors
#         quantized_features = F.conv2d(x_norm, y_norm, stride=1)
#         #quantized_features=quantized_features/(torch.sqrt(x_square)*torch.sqrt(y_square))
#         quantized_features = self.norm1(quantized_features)
#         histogram = torch.mean(quantized_features, (2, 3))

#         return histogram

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class Cos_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(Cos_BoF_Pooling, self).__init__()
        
        self.n_codewords = n_codewords
        self.weight = nn.Parameter(torch.rand(n_codewords,in_channels),requires_grad=True)
        
        self.norm1 = LayerNorm(n_codewords)
        #self.norm1 = nn.BatchNorm2d(n_codewords)
        #self.norm2 = nn.LayerNorm(n_codewords, eps=1e-6) # final norm layer
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        x = normalize(x,axis=3)
        y = normalize(self.weight,axis=1)
        
        x = F.linear(x, y)
        x = self.norm1(x)

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x


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
        self.bofpool = Cos_BoF_Pooling(512,n_codewords)
        self.fc = nn.Linear(n_codewords, num_class)

        
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

def vgg16_bn_cosbof_512():
    return VGG(make_layers(cfg['D'], batch_norm=True),n_codewords=512)
def vgg16_bn_cosbof_256():
    return VGG(make_layers(cfg['D'], batch_norm=True),n_codewords=256)
def vgg16_bn_cosbof_128():
    return VGG(make_layers(cfg['D'], batch_norm=True),n_codewords=128)
def vgg16_bn_cosbof_64():
    return VGG(make_layers(cfg['D'], batch_norm=True),n_codewords=64)

# def vgg19_bn_bof():
#     return VGG(make_layers(cfg['E'], batch_norm=True))