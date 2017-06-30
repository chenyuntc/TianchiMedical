#coding:utf8
import torch as t
from torch import nn
from .module import Module,Flat


class MutltiCNN(Module):
    '''
    <Multilevel Contextual 3-D CNNs for False Positive Reduction \
    in Pulmonary Nodule Detection>
    http://ieeexplore.ieee.org/document/7576695/
    '''
    def __init__(self):
        super(MutltiCNN,self).__init__()
        self.model_name='mutiliconv'
        self.arch1 = nn.Sequential(
            nn.Conv3d(1,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(2,2,2),2),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(4,4,4)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            Flat(),
            nn.Linear(1*1*64,150),
            nn.ReLU(),
            nn.Linear(150,2)
        
        )
        
        self.arch2 = nn.Sequential(
            nn.Conv3d(1,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(2,2,2),2),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(2,2,2)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            Flat(),
            nn.Linear(4*4*4*64,250),
            nn.ReLU(),
            nn.Linear(250,2)
        )
                
        self.arch3 = nn.Sequential(
            nn.Conv3d(1,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(2,2,2),2),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(2,2,2)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            nn.Conv3d(64,64,(5,5,5)),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            
            Flat(),
            nn.Linear(9*9*9*64,250),
            nn.ReLU(),
            nn.Linear(250,2)
        
        )
    def forward(self,x):
        '''
        !TODO: modify this,不必每次都切割,随机,太慢
        '''
       
        hw=24 #输入的大小
        rands=[0 for i in range(10)]
        if self.training == False:rands=[0 for i in range(10)]
        x1 = x[:,:,hw-10+rands[0]:hw+10+rands[0],hw-10+rands[1]:hw+10+rands[1],hw-10+rands[2]:hw+10+rands[2]]
        x2 = x[:,:,hw-15+rands[3]:hw+15+rands[3],hw-15+rands[4]:hw+15+rands[4],hw-15+rands[5]:hw+15+rands[5]]
        x3 = x[:,:,hw-20+rands[6]:hw+20+rands[6],hw-20+rands[4]:hw+20+rands[4],hw-20+rands[5]:hw+20+rands[5]]
        return self.arch1(x1),self.arch2(x2),self.arch3(x3)
    