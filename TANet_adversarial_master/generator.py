import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
# import torchvision.ops as torchops
from collections import OrderedDict
import math
from torch.autograd import Variable
from ops import * 
import pdb 

from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

from resnet import resnet18 
import numpy as np
import cv2 
import pdb 


def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_deconv_layers(cfg):
    layers = []
    in_channels = 1792+512 
    for v in cfg:
        if v == 'U':
            layers += [nn.Upsample(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}

class _ConvNd(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        
    
    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1. / math.sqrt(n)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = ('{name}({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class AdaptiveConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(AdaptiveConv2d, self).__init__(
                in_channels, out_channels, kernel_size, stride, padding, dilation,
                False, _pair(0), groups, bias)

    def forward(self, input, dynamic_weight):
        # Get batch num
        batch_num = input.size(0)

        # Reshape input tensor from size (N, C, H, W) to (1, N*C, H, W)
        input = input.view(1, -1, input.size(2), input.size(3))

        # Reshape dynamic_weight tensor from size (N, C, H, W) to (1, N*C, H, W)
        dynamic_weight = dynamic_weight.view(-1, 1, dynamic_weight.size(2), dynamic_weight.size(3))

        # Do convolution
        conv_rlt = F.conv2d(input, dynamic_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

        # Reshape conv_rlt tensor from (1, N*C, H, W) to (N, C, H, W)
        conv_rlt = conv_rlt.view(batch_num, -1, conv_rlt.size(2), conv_rlt.size(3))

        return conv_rlt


def encoder():
    return make_conv_layers(cfg['E'])

def decoder():
    return make_deconv_layers(cfg['D'])




class C3D(nn.Module):
    def __init__(self):
        super(C3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        # pdb.set_trace()
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        # h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        # h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))
        # #pdb.set_trace()
        h = self.pool5(h)

        return h

        # pdb.set_trace() 

        # h = h.view(-1, 41472)
        # h = self.relu(self.fc6(h))
        # h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(logits)

        # return probs



class ca2_Generator(nn.Module):
    def __init__(self):
        super(ca2_Generator, self).__init__()
        self.encoder = resnet18()
        self.decoder = decoder()

        self.mymodules = nn.ModuleList([
            deconv2d(64, 1, kernel_size=1, padding = 0),
            nn.Sigmoid()
        ])

        self.motionModel = C3D() 
        # self.motionModel.load_state_dict(torch.load('./c3d.pickle')) 
        self.motionModel.cuda() 


    def forward(self, x, targetObject_img, batch_imgClip):      
        x_1, x_2, x_3 = self.encoder(x) 
        #### [10, 128, 38, 38], [10, 256, 19, 19], [10, 256, 10, 10] 
        targetfeat1, targetfeat2, targetfeat3 = self.encoder(targetObject_img) 

        batch_imgClip = torch.transpose(batch_imgClip, 1, 2)
        motionFeats = self.motionModel(batch_imgClip)   ## torch.Size([10, 512, 1, 38, 38])
        motionFeats = torch.squeeze(motionFeats, 2) 
        # pdb.set_trace() 

        fused1 = torch.cat((x_1, targetfeat1), 1)
        fused2 = torch.cat((x_2, targetfeat2), 1)
        fused3 = torch.cat((x_3, targetfeat3), 1)

        fused1 = nn.functional.interpolate(fused1, size=[fused2.shape[2], fused2.shape[3]])  
        fused3 = nn.functional.interpolate(fused3, size=[fused2.shape[2], fused2.shape[3]]) 
        motionFeats = nn.functional.interpolate(motionFeats, size=[fused2.shape[2], fused2.shape[3]]) 

        fuseTemp = torch.cat((fused1, fused2), 1)
        fuseTemp = torch.cat((fuseTemp, fused3), 1)
        fuseTemp = torch.cat((fuseTemp, motionFeats), 1)

        output = self.decoder(fuseTemp)
        output = self.mymodules[0](output)
        output = self.mymodules[1](output) 
        output = nn.functional.interpolate(output, size=[300, 300]) 

        return output 




class App_Discriminator(nn.Module):
    def __init__(self):
        super(App_Discriminator, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(4, 96, kernel_size=7, stride=2),
                                    nn.ReLU(inplace=True), nn.LocalResponseNorm(2), nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 256, kernel_size=5, stride=2),
                                    nn.ReLU(inplace=True), nn.LocalResponseNorm(2), nn.MaxPool2d(kernel_size=3, stride=2))
        self.conv3 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1), nn.ReLU(inplace=True)) 
        self.fc4 = nn.Sequential(nn.Linear(57600, 256), nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 1), nn.ReLU(inplace=True))

    def forward(self, currImage, mask):       

        img_mask = torch.cat((currImage, mask), dim=1) 

        temp_output = self.conv1(img_mask) 
        temp_output = self.conv2(temp_output) 
        temp_output = self.conv3(temp_output) 
        temp_output = temp_output.view(temp_output.size(0), -1)

        # pdb.set_trace() 
        temp_output = self.fc4(temp_output) 
        temp_output = self.fc5(temp_output) 

        return temp_output 



class Dis_C3D(nn.Module):
    def __init__(self):
        super(Dis_C3D, self).__init__()
        
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(1, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, 487)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.pool1(h)

        h = self.relu(self.conv2(h))
        h = self.pool2(h)

        h = self.relu(self.conv3a(h))
        h = self.relu(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu(self.conv4a(h))
        h = self.relu(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu(self.conv5a(h))
        h = self.relu(self.conv5b(h))

        # #pdb.set_trace()
        # h = self.pool5(h)

        return h

        # pdb.set_trace() 

        # h = h.view(-1, 41472)
        # h = self.relu(self.fc6(h))
        # h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        # h = self.dropout(h)

        # logits = self.fc8(h)
        # probs = self.softmax(logits)

        # return probs


class Motion_Discriminator(nn.Module):
    def __init__(self):
        super(Motion_Discriminator, self).__init__()

        self.motionModel = Dis_C3D() 
        self.motionModel.load_state_dict(torch.load('./c3d.pickle')) 
        self.motionModel.cuda() 

        self.conv_2D = nn.Sequential(nn.Conv2d(512, 100, kernel_size=3, stride=2),
                nn.ReLU(inplace=True), nn.LocalResponseNorm(2), nn.MaxPool2d(kernel_size=3, stride=2))

        self.fc4 = nn.Sequential(nn.Linear(900, 256), nn.ReLU(inplace=True))
        self.fc5 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 1), nn.ReLU(inplace=True))

    def forward(self, imgClip, maskClip):       
        #### img_mask_Clip torch.Size([batchSize, 6, 3, 300, 300]) 

        # pdb.set_trace() 
        img_mask_Clip = torch.cat((imgClip, maskClip), dim=2)
        maskMotion_feat = self.motionModel(img_mask_Clip)
        maskMotion_feat = torch.squeeze(maskMotion_feat, dim=2) 
        maskMotion_feat = self.conv_2D(maskMotion_feat) 
        # pdb.set_trace() 

        temp_output = maskMotion_feat.view(maskMotion_feat.size(0), -1)

        # pdb.set_trace() 
        temp_output = self.fc4(temp_output) 
        temp_output = self.fc5(temp_output) 



        return temp_output 


