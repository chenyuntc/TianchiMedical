#coding:utf8
import torch as t
import time


class Module(t.nn.Module):
    '''
    封装了nn.Module,主要是提供了save和load两个方法
    '''

    def __init__(self):
        super(Module,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        if name is None:
            prefix = 'checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    '''
    把输入reshape成（batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
