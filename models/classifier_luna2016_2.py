#coding:utf8
import torch as t
from torch import nn
from .module import Flat, Module

class Luna20162(Module):
    '''
    @refer https://luna16.grand-challenge.org/serve/public_html/pdfs/1706.04303.pdf/
    是否应该有dropout3d?
    '''
    def __init__(self,with_bathnorm=True):
        '''
         @param: with_batchnorm 是否带有batchnorm
        '''

        super(Luna20162,self).__init__()
        self.model_name='L2'
        
        if with_bathnorm:
            self.model=nn.Sequential(
                nn.Conv3d(1,32,(3,3,3)),   
                nn.BatchNorm3d(32),
                nn.ReLU(True),
                nn.Conv3d(32,32,(3,3,3)),
                nn.BatchNorm3d(32),
                nn.ReLU(True),
                nn.MaxPool3d((1,2,2)),
                nn.Dropout3d(),

                nn.Conv3d(32,64,(3,3,3)),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.Conv3d(64,64,(3,3,3)),
                nn.BatchNorm3d(64),
                nn.ReLU(True),
                nn.MaxPool3d((2,2,2)),
                nn.Dropout3d(),

                nn.Conv3d(64,128,(3,3,3)),
                nn.BatchNorm3d(128),
                nn.ReLU(True),
                nn.Conv3d(128,128,(3,3,3)),
                nn.BatchNorm3d(128),
                nn.ReLU(True),
                # nn.MaxPool3d((2,2,2)),
                # nn.Dropout3d(),
                
                Flat(),
                
                nn.Linear(8*128,1024),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(1024,512),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(512,2),    
        )
        else:
            self.model = nn.Sequential(
            nn.Conv3d(1,32,(3,3,3)),   
            nn.ReLU(True),
            nn.Conv3d(32,32,(3,3,3)),
            nn.ReLU(True),
            nn.MaxPool3d((2,2,1)),
            nn.Dropout3d(),

            nn.Conv3d(32,64,(3,3,3)),
            nn.ReLU(True),
            nn.Conv3d(64,64,(3,3,3)),
            nn.ReLU(True),
            nn.MaxPool3d((2,2,2)),
            nn.Dropout3d(),

            nn.Conv3d(64,128,(3,3,3)),
            nn.ReLU(True),
            nn.Conv3d(128,128,(3,3,3)),
            nn.ReLU(True),
            nn.MaxPool3d((2,2,2)),
            nn.Dropout3d(),
            
            Flat(),
            
            nn.Linear(128,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512,2),    
      )
    def forward(self,input):
        return self.model(input)


if __name__ == '__main__':
    input = t.autograd.Variable(t.randn(1,1,36,36,20))
    print(Luna2016()(input))
    print(Luna2016(False)(input))
