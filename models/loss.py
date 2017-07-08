# coding:utf8
import torch
from torch.nn import init
import torch as t
from torch import nn
import time
def loss_all(x, target, x2=None, lamda=1e-5):
    if x2:
        cross_entropy_loss = classifier_loss(x2, target)
    loss1 = dice_loss(x, target)
    # loss2 = torch.sum((1-target)*x*0.01)/(0.01+target.numel()*0.01)
    # loss = loss1 + loss2
    loss2 = ls_loss(x, target)
    return loss1, loss2, cross_entropy_loss, mask


def classifier_loss(x, target):
    x = x.squeeze()
    cross_entropy_loss = torch.nn.functional.cross_entropy(
         x,  torch.autograd.Variable(target), weight=torch.Tensor([1, 1]).cuda())
    return cross_entropy_loss


def dice_loss(x, target):
    # num,loss=0,0
    # for x_,t_ in zip(x,target):
    #     if t_.sum()>40 :
    #         num+=1
    #         loss+=2*(torch.sum(x_*t_))/(torch.sum(x_)+torch.sum(t_))
    # # mask =mask.view(-1,1,1,1,1).expand_as(x)
    batch_size = x.size(0)
    andN = 2 * (torch.sum((x * target).view(batch_size, -1), 1))
    orU1 = torch.sum(x.view(batch_size, -1), 1)
    orU2 = torch.sum(target.view(batch_size, -1), 1)
    dices =  andN / (orU1 + orU2 )
    losses = 1 - dices
    return losses.mean(), (andN, orU1, orU2)


def l2_loss(x, target, weight=None):
    # l1_loss = torch.nn.L1Loss()
    # o = l1_loss(x, target)
    o = (x - target)**2
    if weight:

        mask = target.data
        weight_mask = mask * weight
        mask = torch.autograd.Variable((1 - mask) + weight_mask)
        return (o * mask).mean()

    return o.mean()

def loss4multi(inputs,target):
    # i1,i2,i3 = inputs
    return sum(classifier_loss(ii, target) for ii in inputs)/3.