#encoding:utf-8
from __future__ import print_function
from .module import Module,Flat
import torch as t
import time
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from .layers import Inception_v1,Inception_v2,BasicConv,SingleConv
class Classifier(Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.model_name="classifier"
        self.conv1=BasicConv(1,32)#(40，40，40)
        self.downsample1=Inception_v1(32,32) #(20,20,20)
        self.conv2=BasicConv(32,64)#(20,20,20)
        self.downsample2=Inception_v1(64,64)#(10,10,10)
        self.conv3=BasicConv(64,64)#(10,10,10)
        self.downsample3=Inception_v1(64,64)#(5,5,5)
        self.conv4=BasicConv(64,64)#((5,5,5)
        self.downsample4= nn.Sequential(nn.Conv3d(64,64,(3,3,3)),
                                        nn.BatchNorm3d(64),
                                        nn.ReLU(True),
                                        Flat())#3,3,3
        self.out=nn.Sequential(
            nn.Linear(3*3*3*64,150),
            nn.ReLU(),
            nn.Linear(150,2))
    def forward(self,x):
        conv1=self.conv1(x)
        down1=self.downsample1(conv1)#20
        conv2=self.conv2(down1)
        down2=self.downsample2(conv2)#10
        conv3=self.conv3(down2)
        down3=self.downsample3(conv3)#5
        conv4=self.conv4(down3)
        down4=self.downsample4(conv4)#3
        out=self.out(down4)
        return out
        
        
        
        
        