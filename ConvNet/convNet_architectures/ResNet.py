import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F 
from typing import Type, Any, Callable, Union, List, Optional

from mup import MuReadout

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1): 
    """3X3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, 
                    padding= dilation, groups= groups, bias=False, dilation=dilation
                    )

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class AttentionPool2d(nn.Module): 
    def __init__(self, spacial_dim: int, embed_dim: int, m_transfer: bool, num_heads: int, output_dim: int = None,): 
        super().__init__()
        if m_transfer:
            self.positional_embedding= nn.Parameter(torch.randn(spacial_dim **2 +1, embed_dim)/ embed_dim **0.5)
            self.k_proj = MuReadout(embed_dim, embed_dim, readout_zero_init=True)
            self.q_proj = MuReadout(embed_dim, embed_dim, readout_zero_init=True)
            self.v_proj = MuReadout(embed_dim, embed_dim, readout_zero_init=True)
            self.c_proj = MuReadout(embed_dim, output_dim or embed_dim, readout_zero_init=True)
            self.num_heads= num_heads

        else: 
            self.positional_embedding= nn.Parameter(torch.randn(spacial_dim **2 +1, embed_dim)/ embed_dim **0.5)
            self.k_proj=nn.Linear(embed_dim, embed_dim)
            self.q_proj= nn.Linear(embed_dim, embed_dim)
            self.v_proj= nn.Linear(embed_dim, embed_dim)
            self.c_proj= nn.Linear(embed_dim, output_dim or embed_dim)
            self.num_heads= num_heads

    def forward(self, x): 
        x= x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]).permute(2, 0, 1) # NCHW --> (HW)NC
        x= torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # (HW+1)NC
        x= x + self.positional_embedding[:, None, :].to(x.dtype) #(HW+1)NC
        x, _ = F.multi_head_attention_forward(query=x, key=x, value=x, embed_dim_to_check=x.shape[-1],num_heads= self.num_heads, 
            q_proj_weight= self.q_proj.weight, 
            k_proj_weight= self.k_proj.weight, 
            v_proj_weight= self.v_proj.weight, 
            in_proj_weight=None, 
            in_proj_bias= torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]), 
            bias_k= None, 
            bias_v= None,
            add_zero_attn=False, 
            dropout_p= 0, 
            out_proj_weight= self.c_proj.weight, 
            out_proj_bias = self.c_proj.bias, 
            use_separate_proj_weight=True, 
            training= self.training, 
            need_weights= False
        )
        return x[0]

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
    expansion=4 
    __constants__= ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, 
                base_width=64, dilation=1, norm_layer=None
    ): 
        super(Bottleneck, self).__init__()
        self.downsample = downsample 
        if norm_layer is None: 
            norm_layer= nn.BatchNorm2d 
        
        width= int(planes* (base_width/ 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1= norm_layer(width)
        self.conv2= conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes* self.expansion)
        self.bn3= norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace= True)

        self.stride = stride 

    def forward(self, x): 
        identity= x 
        
        out= self.conv1(x)
        out= self.bn1(out)
        out= self.relu(out) 

        out= self.conv2(out)
        out= self.bn2(out) 
        out= self.relu(out) 

        out= self.conv3(out)
        out= self.bn3(out) 

        if self.downsample is not None: 
            identity= self.downsample(x)

        out += identity 
        out= self.relu(out)
        return out

class ResNet(nn.Module): 

    def __init__(self, block, layers, num_classes=1000, zero_init_residual= False, 
                groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 width_mult=1,feat_scale=1, m_transfer=True,  pooling_type="standard",image_resolution=224,num_heads=2, output_dim=2048
    ): 

        super(ResNet, self).__init__()

        if norm_layer is None: 
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64 * width_mult
        self.dilation = 1
        self.pooling_type=pooling_type 
        if self.pooling_type=="attention_pooling": 
            print("ATTENTION You implement Attention pooling method")
        self.m_transfer= m_transfer
        

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        ## First Block of ResNet model
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## Connect Basic block and Bottelneck block
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(block, 128 * width_mult, layers[1], stride=2,
                                        dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256 * width_mult, layers[2], stride=2,
                                        dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512 * width_mult, layers[3], stride=2,
                                        dilate=replace_stride_with_dilation[2])
    
        ## Standard Pytorch Linear layer
        self.fc = nn.Linear(feat_scale*512 * block.expansion* width_mult, num_classes)
        ##m_transfer linear layer
        self.m_transfer_fc=MuReadout(feat_scale*512 * block.expansion * width_mult, num_classes, readout_zero_init=True)
        ## Standard AveragePooling2D 
        self.avgpool= nn.AdaptiveAvgPool2d((1, 1))
        ## Attention pooling layer
        embed_dim = self.base_width * 32  # the ResNet feature dimension
        self.attentionpool= AttentionPool2d(image_resolution//32, embed_dim, num_heads, output_dim)


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False): 
        norm_layer= self._norm_layer
        downsample=None 
        previous_dilation= self.dilation
        if dilate: 
            self.dilation *=stride 
            stride= 1 
        if stride !=1 or self.inplanes !=planes *block.expansion: 
            downsample= nn.Sequential(conv1x1(self.inplanes, planes*block.expansion, stride), 
            norm_layer(planes*block.expansion),
            )
        layers=[]
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups, 
                self.base_width, previous_dilation, norm_layer)) 

        self.inplanes = planes*block.expansion

        for _ in range(1, blocks): 
            layers.append(block(self.inplanes, planes, groups=self.groups, 
            base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        
        return nn.Sequential(*layers)

    def _forward_impl(self, x): 
        x= self.conv1(x)
        x= self.bn1(x)
        x= self.relu(x)
        x= self.maxpool(x)

        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x= self.layer4(x)
        if self.pooling_type=="attention_pooling":
             
            x= self.attention_pooling(x)
        else: 
            x= self.avgpool(x)
            x = torch.flatten(x, 1)
        if self.m_transfer: 
            x=self.m_transfer_fc(x)
        else: 
            
            x=self.fc(x)
        return x 
    
    def forward(self, x): 
        return self._forward_impl(x)

class ResNet_v1(nn.Module): 
    pass 

def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

