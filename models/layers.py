#coding:utf8
'''
常用的层,比如inception block,residual block
'''

#coding:utf8
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable

class Deconv(nn.Module):
    def __init__(self,cin,cout):
        super(Deconv, self).__init__()
        self.model=nn.Sequential(OrderedDict([
            ('deconv1',nn.ConvTranspose3d(cin,cout,2,stride=2)),            
            ('norm', nn.BatchNorm3d(cout)),
            ('relu', nn.ReLU(inplace=True)),        
            ])) 
    def forward(self,x):
        return self.model(x)
class SingleConv(nn.Module):
    def __init__(self,cin,cout,padding=1):
        super(SingleConv, self).__init__()
        self.padding=padding
        self.model=nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin,cout,3,padding=self.padding)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),        
            ])) 
    def forward(self,x):
        return self.model(x)
class BasicConv(nn.Module):
    def __init__(self,cin,cout):
        super(BasicConv, self).__init__()
        self.model=nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv3d(cin,cout,3,padding=1)),
            ('norm1_1', nn.BatchNorm3d(cout)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv3d(cout,cout,3,padding=1)),
            ('norm1_2', nn.BatchNorm3d(cout)),
            ('relu1_2', nn.ReLU(inplace=True)),         
            ])) 
    def forward(self,x):
        return self.model(x)
    
class Inception_v1(nn.Module):
    def __init__(self,cin,co,relu=True,norm=True):
        super(Inception_v1, self).__init__()
        assert(co%4==0)
        cos=[co/4]*4
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm3d(co))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        
        self.branch1 = nn.Conv3d(cin, cos[0], 1,stride=2)
        
        self.branch2= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin,2*cos[1], 1)),
            ('norm1', nn.BatchNorm3d(2*cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2*cos[1], cos[1], 3,stride=2,padding=1)),
            ])) 
        self.branch3= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, 2*cos[2], 1,stride=1)),
            ('norm1', nn.BatchNorm3d(2*cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2*cos[2], cos[2], 5,stride=2,padding=2)),
            ])) 

        self.branch4=nn.Sequential(OrderedDict([
            ('pool',nn.MaxPool3d(2)),
            ('conv',nn.Conv3d(cin, cos[3], 1,stride=1))
                ]))

    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=torch.cat((branch1,branch2,branch3,branch4),1)
        return self.activa(result)


    
class Inception_v2(nn.Module):
    def __init__(self,cin,co,relu=True,norm=True):
        super(Inception_v2, self).__init__()
        assert(co%4==0)
        cos=[co/4]*4
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm3d(co))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        
        self.branch1 = nn.Conv3d(cin, cos[0], 1)
        
        self.branch2= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin,2*cos[1], 1)),
            ('norm1', nn.BatchNorm3d(2*cos[1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2*cos[1], cos[1], 3,stride=1,padding=1)),
            ])) 
        self.branch3= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin, 2*cos[2], 1,stride=1)),
            ('norm1', nn.BatchNorm3d(2*cos[2])),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(2*cos[2], cos[2], 5,stride=1,padding=2)),
            ])) 

        self.branch4=nn.Sequential(OrderedDict([
            ('pool',nn.MaxPool3d(3,stride=1,padding=1)),
            ('conv',nn.Conv3d(cin, cos[3], 1,stride=1))
                ]))

    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        branch4=self.branch4(x)
        result=torch.cat((branch1,branch2,branch3,branch4),1)
        return self.activa(result)
class res_conc_block(nn.Module):
    #'''
    #残差链接模块
    #分支1：3*3，stride=1的卷积
    #分支2:1*1，stride=1的卷积，3*3，stride=1的卷积
    #分支3：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=1的卷积
    #分支1,2,3concat到一起，1*1，stride=1卷积
    #最后在与input相加
    #'''
    def __init__(self,cin,cn,norm=True,relu=True):
        super(res_conc_block,self).__init__()
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm3d(3*cn))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        self.branch1=nn.Conv3d(cin, cn, 3,padding=1)
        self.branch2= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin,cn, 1)),
            ('norm1', nn.BatchNorm3d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cn, cn, 3,stride=1,padding=1)),
            ]))
        self.branch3= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin,cn, 1)),
            ('norm1', nn.BatchNorm3d(cn)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(cn, cn, 3,stride=1,padding=1)),
            ('norm2', nn.BatchNorm3d(cn)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv3', nn.Conv3d(cn, cn, 3,stride=1,padding=1)),
            ]))
        self.merge=nn.Conv3d(3*cn,cin,1,1)
    def forward(self,x):
        branch1=self.branch1(x)
        branch2=self.branch2(x)
        branch3=self.branch3(x)
        result=torch.cat((branch1,branch2,branch3),1)
        result=self.activa(result)
        return x+self.merge(result)
        

class feat_red(nn.Module):
    #'''
    #特征压缩模块，经过此模块后空间大小不变，通道数减半
    #'''
    def __init__(self,cin,co,relu=True,norm=True):
        super(feat_red,self).__init__()
        assert(cin/co==2)
        self.model=nn.Sequential(OrderedDict([
            ('conv',nn.Conv3d(cin, co, 1,stride=1))
                ]))
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm3d(co))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        
         
        self.conv1=nn.Conv3d(cin,co,1,stride=1)
    def forward(self,x):
        result=self.model(x)
        
        return self.activa(result)
               
class spatial_red_block(nn.Module):
    #空间压缩模块，经过此模块后input的大小减半，通道数加倍
    # 分支1：2*2，stride=2的最大池化
    # 分支2:3*3，stride=2的卷积
    # 分支3:1*1，stride=1的卷积，3*3，stride=2的卷积
    # 分支4：1*1，stride=1的卷积，3*3，stride=1的卷积，3*3，stride=2的卷积
    def __init__(self,cin,relu=True,norm=True):
        super(spatial_red_block,self).__init__()
        #super(spatial_red_block,self).__init__()
        co=cin       
        assert(cin%16==0),'the input channel must be divided by 16'
        self.maxpool= nn.MaxPool3d(2) 
        self.activa=nn.Sequential()
        if norm:self.activa.add_module('norm',nn.BatchNorm3d(2*cin))
        if relu:self.activa.add_module('relu',nn.ReLU(True))
        self.branch2_conv1=nn.Conv3d(cin,co/4,3,stride=2,padding=1)
        
        self.branch3= nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(cin,co/4,1)),
            ('norm1', nn.BatchNorm3d(co/4)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv3d(co/4,5*co/16,3,stride=2,padding=1)),
            ]))       
        self.branch4= nn.Sequential(OrderedDict([
            ('conv1',nn.Conv3d(cin,co/4,1)),
            ('norm1',nn.BatchNorm3d(co/4)),
            ('relu1',nn.ReLU(inplace=True)),
            ('conv2',nn.Conv3d(co/4,5*co/16,3,stride=1,padding=1)),
            ('norm2',nn.BatchNorm3d(5*co/16)),
            ('relu2',nn.ReLU(inplace=True)),
            ('conv3',nn.Conv3d(5*co/16,7*co/16,3,stride=2,padding=1)),
            
                        
            ]))
               
    def forward(self,x):
        result_1=self.maxpool(x)
        
        result_2=self.branch2_conv1(x)
               
        result_3=self.branch3(x)
    
        result_4=self.branch4(x)        
        result=torch.cat((result_1,result_2,result_3,result_4),1)
        return self.activa(result)
