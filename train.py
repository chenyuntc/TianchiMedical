#coding:utf8
import importlib
from config import opt
import numpy as np
import fire,time
import torchnet as tnt
import os
import torch as t
from data.dataset import ClsDataset, SegDataset
from utils.util import get_optimizer
from utils.visualize import Visualizer
import models
vis = Visualizer(opt.env)
from models import loss as Loss_
# t.backend.cudnn.benchmark = True

def parse(kwargs):
    ## 处理配置和参数
    for k,v in kwargs.iteritems():
        if not hasattr(opt,k):
            print("Warning: opt has not attribut %s" %k)
        setattr(opt,k,v)
    for k,v in opt.__class__.__dict__.iteritems():
        if not k.startswith('__'):print(k,getattr(opt,k))

    vis.reinit(opt.env)

def train_seg(**kwargs):
    '''
    训练分割网络
    '''
    parse(kwargs)

    loss_function = getattr(Loss_,opt.seg_loss_function)
    model = getattr(models,opt.seg_model)().cuda() 
    if opt.seg_model_path is not None:
        model.load(opt.seg_model_path)
    dataset = SegDataset()
    dataloader = t.utils.data.DataLoader(dataset,opt.batch_size,       
                        num_workers=opt.num_workers,
                        shuffle=opt.shuffle,
                        pin_memory=opt.pin_memory)

    pre_loss= 100
    lr = opt.lr
    optimizer = get_optimizer(model,opt.lr)
    loss_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.max_epoch):

        loss_meter.reset()
        start=time.time()

        for ii, (input, mask) in enumerate(dataloader):

            optimizer.zero_grad()
            input = t.autograd.Variable(input).cuda()  
            target = t.autograd.Variable(mask).cuda()  

            output = model(input)
            loss, _ = loss_function(output, target)
            # othter_info = [jj.data.cpu().tolist() for jj in othter_info]
            # vis.vis.text(othter_info, win='othter_info')

            loss_meter.add(loss.data[0])
            (loss).backward()
            optimizer.step()
            

            ### 可视化， 记录， log，print
            if ii %opt.plot_every== 0 and ii>opt.plot_every:
                if os.path.exists(opt.seg_debug_file):
                    import ipdb; ipdb.set_trace()
                vis_plots = {'loss':loss_meter.value()[0],'ii':ii}
                vis.plot_many(vis_plots)

                # 随机展示一张图片
                k = t.randperm(input.size(0))[0]
                vis.vis.histogram(
                    output.data[k].view(-1).cpu(), win=u'output_hist', opts=dict
                    (title='output_hist'))
                #！TODO: tell 代成 make 1/3 和1 ，而不是1和3
                vis_imgs = {'input':input.data[k],'mask':target.data[k],'output':output.data[k]}
                vis.img_grid_many(vis_imgs)
                                
                print "epoch:%4d/%4d,time: %.8f,loss: %.8f " %(epoch,ii,time.time()-start,loss_meter.value()[0])

        model.save() 
        vis.log({' epoch:':epoch,' loss:':str(loss_meter.value()[0]),' lr: ':lr})
                
        # info = time.strftime('[%m%d %H:%M] epoch') + str(epoch) + ':' + \
        #     str(loss_meter.value()[0]) + str('; lr:') + str(self.lr) + '<br>'
        # vis.vis.texts += info
        # vis.vis.text(vis.vis.texts, win=u'log')

        # 梯度衰减
        if loss_meter.value()[0] > pre_loss:
            lr = lr*opt.lr_decay
            optimizer = get_optimizer(model, lr)


        pre_loss = loss_meter.value()[0]
        if lr < opt.min_lr:
            break
        

def val_cls(model,loss_function):

    model.eval()
    dataset = ClsDataset(val=True)
    dataloader = t.utils.data.DataLoader(dataset,
                        opt.batch_size,
                        num_workers=opt.num_workers,
                        shuffle=False,pin_memory=opt.pin_memory)
    
    confusem  = tnt.meter.ConfusionMeter(2)
    loss_meter = tnt.meter.AverageValueMeter()

    for ii, (input, label) in enumerate(dataloader):

        input = t.autograd.Variable(input,volatile=True).cuda()
        target = label.cuda()
        output = model(input)
        loss = loss_function(output, target)
        confusem.add(output.data, target)
        loss_meter.add(loss.data[0])

    
    model.train()
    return confusem,loss_meter



def train_cls(**kwargs):
    '''
    训练分类网络
    '''
    parse(kwargs)

    loss_function = getattr(Loss_,opt.cls_loss_function)
    model = getattr(models,opt.cls_model)().cuda()
    if opt.cls_model_path is not None:
        model.load(opt.cls_model_path)
    dataset = ClsDataset()
    dataloader = t.utils.data.DataLoader(dataset,opt.batch_size,       
                        num_workers=opt.num_workers,
                        shuffle=opt.shuffle,
                        pin_memory=opt.pin_memory)


    pre_loss= 100
    lr = opt.lr
    optimizer = get_optimizer(model,opt.lr)
    loss_meter = tnt.meter.AverageValueMeter()

    confusem = tnt.meter.ConfusionMeter(2)
    for epoch in range( opt.max_epoch):
        loss_meter.reset()
        confusem.reset()
        start=time.time()
        for ii, (input, label) in enumerate(dataloader):

            optimizer.zero_grad()
            input = t.autograd.Variable(input).cuda()
            #!TODO: modify label 
            target = label.cuda()
            #!TODO: output maybe a list  
            output = model(input)
            loss = loss_function(output, target)
            (loss).backward()
            optimizer.step()



            # loss1,loss2,loss3 = loss_function(score1,target),loss_function(score2,target),loss_function(score3,target)
            # loss = loss1+loss2+loss3
            # prob1,prob2,prob3=t.nn.functional.softmax(score1),t.nn.functional.softmax(score2),t.nn.functional.softmax(score3)
            # prob=(prob1+prob2+prob3)/3.0
            confusem.add(output.data, target)
            loss_meter.add(loss.data[0])


            if ii % opt.plot_every == 0 and ii > 0:    


                vis_plots = {'loss':loss_meter.value()[0],'ii':ii}
                vis.plot_many(vis_plots)

                
                vis.vis.text('cm:%s, loss:%s' % (
                    str(confusem.value()), loss.data[0]), win=u'confusionmatrix')
                if os.path.exists(opt.cls_debug_file):
                    import ipdb
                    ipdb.set_trace()

                # print "epoch:%4d,time:%.8f,loss:%.8f" %(epoch,time.time()-start, loss_meter.value()[0])
                    
    
        model.save()
        
        # info = time.strftime('[%m%d %H:%M] epoch') + str(epoch) + ':' + \
        #     str(loss_meter.value()[0]) + str('; lr:') + str(lr) + '<br>'
        # vis.vis.texts += info
        # vis.vis.text(vis.vis.texts, win=u'log')
        val_cm,val_loss = val_cls(model,loss_function)
        # vis.log(   {'epoch:':epoch,\
        #             'loss:':str(loss_meter.value()[0]),\
        #             'lr:':lr,\
        #             'cm:':str(confusem.value()),\
        #             'val_loss':str(val_loss.value()[0]),\
        #             'val_cm':str(val_cm.value())
        #             })
        vis.log('epoch:{epoch},loss:{loss:.4f},lr:{lr:.6f},cm:{cm},val_loss:{val_loss:.4f},val_cm:{val_cm}'.format(
            epoch = epoch,
            loss =(loss_meter.value()[0]),
            lr = lr,
            cm = str(confusem.value()),
            val_loss = (val_loss.value()[0]),
            val_cm = str(val_cm.value())


        ))

        if val_loss.value()[0] > pre_loss*1.:
            lr = lr * opt.lr_decay
            optimizer = get_optimizer(model, lr)
        pre_loss = val_loss.value()[0]
        if lr < opt.min_lr:
            break



def main(**kwargs):
    '''
    python a.py main --Train=False --datasetp='sdgsg'
    '''

    parse(kwargs)
    
    if opt.cls: train_cls()
    else: train_seg()

def help():
    '''
    打印帮助的信息
    python file.py help
    '''

    print('''
    usage : python file.py <function> [--args=value]
    
    example: 
    python a.py main --Train=False --datasetp='sdgsg'
    python a.py train_cls --dataset='path/to/dataset/root/'
    python a.py train_seg --env='seg_env0701'
    
    avaiable args:
    ''')

    from inspect import getsource

    source = (getsource(opt.__class__))
    print source
    # for line in source.split('\r\n'):
    

if __name__=="__main__":

    fire.Fire()
