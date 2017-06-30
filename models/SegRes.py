#coding:utf8
from __future__ import print_function
from .module import Module
import torch as t
import time
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from .layers import Inception_v1,Inception_v2,BasicConv,Deconv,SingleConv

class Segmentation(Module):
    def __init__(self):
        super(Segmentation,self).__init__()
        self.model_name="seg"
        self.conv1=BasicConv(1,32)#(64，64，64)
        self.downsample1=Inception_v1(32,32) #(32,32,32)
        self.conv2=BasicConv(32,64)#(32,32,32)
        self.downsample2=Inception_v1(64,64)#(16,16,16)
        self.conv3=BasicConv(64,128)#(16,16,16)
        self.downsample3=Inception_v1(128,128) #(8,8,8)
        self.conv4=BasicConv(128,256)#(8,8,8)
        self.downsample4=Inception_v1(256,256)  #(4,4,4) 
        
        self.conv4_=SingleConv(256,128)
        self.incept4=Inception_v2(128,128)
        self.deconv4=Deconv(128,128)
        
        self.conv5=SingleConv(384,128)
        self.incept5=Inception_v2(128,128)
        self.deconv5=Deconv(128,128)
        
        self.conv6=SingleConv(256,64)
        self.incept6=Inception_v2(64,64)
        self.deconv6=Deconv(64,64)
        
        self.conv7=SingleConv(128,32)
        self.incept7=Inception_v2(32,32)
        self.deconv7=Deconv(32,32)
        
        self.conv8=SingleConv(64,32)
        self.incept8=Inception_v2(32,32)
        self.conv9=nn.Conv3d(32,1,1)
        
        self.activate=nn.Sigmoid()
        
        
        
    def forward(self,x):
        conv1=self.conv1(x)#(64，64，64)
        down1=self.downsample1(conv1)#(32,32,32)
        conv2=self.conv2(down1)#(32,32,32)
        down2=self.downsample2(conv2)#(16,16,16)
        conv3=self.conv3(down2)#(16,16,16)
        down3=self.downsample3(conv3)#(8,8,8)
        conv4=self.conv4(down3)#(8,8,8)
        down4=self.downsample4(conv4)#(4,4,4)
        
        conv4_=self.incept4(self.conv4_(down4))
        up4=self.deconv4(conv4_)#(8,8,8)
        up4=t.cat((up4,conv4),1)
        
        conv5=self.incept5(self.conv5(up4))
        up5=self.deconv5(conv5)#(16,16,16)
        up5=t.cat((up5,conv3),1)
        
        conv6=self.incept6(self.conv6(up5))
        up6=self.deconv6(conv6)#(32,32,32)
        up6=t.cat((up6,conv2),1)
        
        conv7=self.incept7(self.conv7(up6))
        up7=self.deconv7(conv7)
        up7=t.cat((up7,conv1),1)
        
        conv8=self.incept8(self.conv8(up7))
        conv9=self.conv9(conv8)
        
        return  self.activate(conv9)
 