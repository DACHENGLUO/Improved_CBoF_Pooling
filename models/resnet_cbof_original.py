from typing import Type, Any, Callable, Union, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import numpy as np
#from ..utils import _log_api_usage_once


__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

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
    
# class Cos_BoF_Pooling(nn.Module):
#     def __init__(self,in_channels: int, n_codewords: int):
#         super(Cos_BoF_Pooling, self).__init__()
#         self.n_codewords = n_codewords
#         self.weight = nn.Parameter(torch.rand(n_codewords,in_channels,1,1),requires_grad=True)
#         #self.norm1 = nn.BatchNorm2d(n_codewords)
#         self.norm1 = LayerNorm(n_codewords)
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
class Cos_BoF_SPP_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(Cos_BoF_SPP_Pooling, self).__init__()
        
        self.n_codewords = n_codewords
        self.weight = nn.Parameter(torch.rand(n_codewords,in_channels),requires_grad=True)
        
        self.norm1 = LayerNorm(n_codewords)
        #self.norm1 = nn.BatchNorm2d(n_codewords)
        #self.norm2 = nn.LayerNorm(n_codewords, eps=1e-6) # final norm layer
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self, x):
        
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        x_norm = normalize(x,axis=3)
        y_norm = normalize(self.weight,axis=1)
        
        x = F.linear(x_norm, y_norm)
        x = self.norm1(x)

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        B, C, H, W = x.shape
        h_mid = int(H/2)
        w_mid = int(W/2)
        histogram1 = torch.mean(x[:, :, :h_mid, :w_mid], (2, 3))
        histogram2 = torch.mean(x[:, :, h_mid:, :w_mid], (2, 3))
        histogram3 = torch.mean(x[:, :, :h_mid, w_mid:], (2, 3))
        histogram4 = torch.mean(x[:, :, h_mid:, w_mid:], (2, 3))
        histogram = torch.mean(x, (2, 3))
    
        return torch.cat((histogram1,histogram2,histogram3,histogram4,histogram),1)    

        

class App_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(App_BoF_Pooling, self).__init__()
        
        self.n_codewords = n_codewords
        self.D = 2000
        self.s_D = np.sqrt(self.D)
        self.weight = nn.Parameter(torch.rand(n_codewords,in_channels),requires_grad=True)
        self.trans = nn.Parameter(torch.rand(self.D,in_channels),requires_grad=False)
        #self.gamma = nn.Parameter(1e-6 * torch.ones((n_codewords)), requires_grad=True) 
        
        
        self.norm1 = LayerNorm(self.D, eps=1e-6)
        self.norm2 = LayerNorm(self.D, eps=1e-6)
        self.norm3 = nn.BatchNorm2d(n_codewords)
        #self.norm2 = nn.LayerNorm(n_codewords, eps=1e-6) # final norm layer
        #nn.init.kaiming_uniform_(self.weight)
        nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self, x):
        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        
        x = 1/self.s_D * torch.cos(F.linear(x, self.trans))
        x = self.norm1(x)
        
        y = 1/self.s_D * torch.cos(F.linear(self.weight, self.trans))
        y = self.norm2(y)
        
        x = F.linear(x, y)
        #x = self.gamma * x
        

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm3(x)
        x = input + x
        x = x.mean([2,3])
        #x = self.norm2(x)
    
        
        return x
    
class SP_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(SP_BoF_Pooling, self).__init__()
        
        self.norm2 = nn.LayerNorm(in_channels+7*7, eps=1e-6) # final norm layer
    

    def forward(self, x):
        x1 = torch.mean(x, (2, 3))
        x2 = torch.mean(x, (1))
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)
        x = torch.cat((x1,x2),dim=1)
        x = self.norm2(x)
        return x
    
class Att_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(Att_BoF_Pooling, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=0, groups=in_channels) 
        #self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()
        
        #self.norm1 = nn.BatchNorm2d(in_channels)
        #self.act = nn.ReLU()
        #self.channelwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) 
        self.norm2 = nn.BatchNorm2d(n_codewords)
        #self.norm2 = nn.LayerNorm(in_channels, eps=1e-6) # final norm layer
        #nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self, x):
        
        #x = self.norm1(x)
        gamma = self.depthwise(x)
        #gamma = self.avg_pool(gamma)
        gamma = self.sigmoid(gamma)
        #x = self.channelwise1(x)
        x = x * gamma.expand_as(x)
        x = self.norm2(x)
        x = x.mean([2,3])
        #x = self.norm2(x)
        return x

class Global_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(Global_BoF_Pooling, self).__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=0, groups=in_channels) 
        #self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=0) 
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.channelwise1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False) 
        self.act = nn.ReLU()
        self.channelwise2 = nn.Conv2d(in_channels, n_codewords, kernel_size=1, bias=False) 
        self.norm2 = nn.BatchNorm2d(n_codewords)
        #self.norm2 = nn.LayerNorm(n_codewords, eps=1e-6) # final norm layer
        #nn.init.kaiming_uniform_(self.weight, mode="fan_out", nonlinearity="relu")
    

    def forward(self, x):
        
        x = self.depthwise(x)
        x = self.norm1(x)
        x = self.channelwise1(x)
        x = self.act(x)
        x = self.channelwise2(x)
        x = self.norm2(x)
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x
    
class Dot_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(Dot_BoF_Pooling, self).__init__()
        
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
        
        #x = x_square + y_square - F.linear(x, self.weight)
        #x = torch.square(self.gamma) * x
        x = F.linear(x, self.weight)
        x = self.gamma * x
        x = F.softmax(x,dim=3)
        

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm1(x)
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x
class Dot_NA_BoF_Pooling(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(Dot_NA_BoF_Pooling, self).__init__()
        
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
        
        #x = x_square + y_square - F.linear(x, self.weight)
        #x = torch.square(self.gamma) * x
        x = F.linear(x, self.weight)
        x = self.gamma * x
        

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm1(x)
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x
    
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
        x = torch.square(self.gamma) * x
        #x = self.gamma * x
        x = F.softmax(-x,dim=3)
        

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm1(x)
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x

class BoF_Pooling_gamma_index(nn.Module):
    def __init__(self,in_channels: int, n_codewords: int):
        super(BoF_Pooling_gamma_index, self).__init__()
        
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
        #x = self.gamma * x
        x = torch.pow(2, self.gamma) * x
        x = F.softmax(-x,dim=3)
        

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = self.norm1(x)
        x = x.mean([2,3])#*self.n_codewords
        #x = self.norm2(x)
        return x    
    
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        BoF_Type: str = 'BoF',
        n_crosswords : int = 0, # 0 : n_crosswords equal to the number of channels in last feature maps
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        #_log_api_usage_once(self)
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        #self.avgpool = BoF_Pooling(512 * block.expansion,512 * block.expansion)
        if n_crosswords == 0:
            n_crosswords = 512 * block.expansion
        self.n_crosswords = n_crosswords
        self.linear_input = self.n_crosswords
        if BoF_Type == 'BoF_cosSPP':
            self.linear_input = self.n_crosswords * 5
        if BoF_Type == 'BoF_SP':
            self.linear_input = n_crosswords + 7*7
        self.avgpool = self._make_pooling_layer(BoF_Type, 512 * block.expansion, self.n_crosswords)
        self.fc = nn.Linear(self.linear_input, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]


    
    def _make_pooling_layer(
        self,
        BoF_Type: str = 'BoF',
        input_channels: int = 512*4,
        n_crosswords : int = 512*4,
    ) -> nn.Sequential:
        layers = []
        if BoF_Type == 'BoF':
            layers.append(BoF_Pooling(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_gamma_index':
            layers.append(BoF_Pooling_gamma_index(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_global':
            layers.append(Global_BoF_Pooling(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_att':
            layers.append(Att_BoF_Pooling(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_app':
            layers.append(App_BoF_Pooling(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_dot':
            layers.append(Dot_BoF_Pooling(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_dot_na':
            layers.append(Dot_NA_BoF_Pooling(input_channels,n_crosswords))
        elif BoF_Type == 'BoF_cos':
            layers.append(Cos_BoF_Pooling(input_channels,n_crosswords))         
        elif BoF_Type == 'BoF_cosSPP':
            layers.append(Cos_BoF_SPP_Pooling(input_channels,n_crosswords))  
        elif BoF_Type == 'BoF_SP':
            layers.append(SP_BoF_Pooling(input_channels,n_crosswords))  
        
            
        
        return nn.Sequential(*layers)

    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    BoF_Type: str,
    n_crosswords: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any,
) -> ResNet:
    model = ResNet(block, layers, BoF_Type, n_crosswords, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet_cbof_original18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet_cbof_original34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)

def resnet_cbofSPP_cos50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_cosSPP', 408, pretrained, progress, **kwargs)

def resnet_cbofSPP_cos101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_cosSPP', 408, pretrained, progress, **kwargs)

def resnet_cbof_SP50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_SP', 0, pretrained, progress, **kwargs)

def resnet_cbof_SP101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_SP', 0, pretrained, progress, **kwargs)


def resnet_cbof_cos50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_cos', 0, pretrained, progress, **kwargs)

def resnet_cbof_cos101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_cos', 0, pretrained, progress, **kwargs)

def resnet_cbof_dot50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_dot', 0, pretrained, progress, **kwargs)

def resnet_cbof_dot101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_dot', 0, pretrained, progress, **kwargs)

def resnet_cbof_dotna50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_dot_na', 0, pretrained, progress, **kwargs)

def resnet_cbof_dotna101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_dot_na', 0, pretrained, progress, **kwargs)

def resnet_cbof_app50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_app', 0, pretrained, progress, **kwargs)

def resnet_cbof_app101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_app', 0, pretrained, progress, **kwargs)

def resnet_cbof_att50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_att', 0, pretrained, progress, **kwargs)

def resnet_cbof_att101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_att', 0, pretrained, progress, **kwargs)

def resnet_cbof_original50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF', 0, pretrained, progress, **kwargs)


def resnet_cbof_original101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF', 0, pretrained, progress, **kwargs)

def resnet_cbof_global50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_global', 0, pretrained, progress, **kwargs)


def resnet_cbof_global101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_global', 0, pretrained, progress, **kwargs)

def resnet_cbof_gamma_index50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], 'BoF_gamma_index', 0,  pretrained, progress, **kwargs)


def resnet_cbof_gamma_index101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet101", Bottleneck, [3, 4, 23, 3], 'BoF_gamma_index', 0 ,pretrained, progress, **kwargs)

def resnet_cbof_original152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet("resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet("resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet("wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
