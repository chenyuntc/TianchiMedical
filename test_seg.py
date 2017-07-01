#encoding:utf-8
import os
import sys
import time
from glob import glob
import fire
import numpy as np
import torch
from skimage import color, data, measure, morphology, segmentation

import torch as t
from common_del.cysb import cropBlocks, get_ct, normalize, resample
from utils.util import  get_optimizer
from utils.visualize import Visualizer
from data.util import check_center
from models.SegRes import Segmentation
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x
class Config:
    seg_size=64#送入分割网络的立方体的大小
    batch_size=4
    img_dir='/home/x/data/datasets/tianchi/train/'#待测试图像路径
    save_dir='/mnt/7/0630_train_no_normalization/'#分割结果保存路径
    model_dir="checkpoints/seg_0527_05:10:49.pth"#预加载模型路径
    prob_threshould=0.8#二值化概率阈值
    crop_size=48#疑似结点切割大小
    is_save=True#是否保存
    
opt = Config() 
def parse(kwargs):
    ## 处理配置和参数
    for k,v in kwargs.iteritems():
        if not hasattr(opt,k):
            print("Warning: opt has not attribut %s" %k)
        setattr(opt,k,v)
    for k,v in opt.__class__.__dict__.iteritems():
        if not k.startswith('__'):print(k,v)
def seg(file_name,model):
    seg_size = opt.seg_size
    print " data prepared................................."
    img_arr,origin,spacing=get_ct(file_name+'.mhd')
    img_new=normalize(img_arr)
    depth, height, width = img_new.shape
    blocks, indexs = cropBlocks(img_new)
    probs = np.zeros(img_new.shape, dtype=np.float32)
    num = np.array(img_new.shape) / seg_size
    off = np.array(img_new.shape) - seg_size * num
    off_min = off / 2
    batch_num=opt.batch_size
    print "doing on patient:", file_name

    for i in range(blocks.shape[0]):
        if (i % batch_num == batch_num - 1):
            batch_inputs_numpy = [torch.from_numpy(blocks[j][np.newaxis, np.newaxis, :, :, :]) for j in range(i - batch_num + 1, i + 1)]
            batch_inputs = torch.autograd.Variable(torch.cat(batch_inputs_numpy, 0), volatile=True).cuda()
            batch_outputs = model(batch_inputs)
            for j in range(i - batch_num + 1, i + 1):
                probs[off_min[0] + indexs[j, 0] * seg_size:off_min[0] + indexs[j, 0] * seg_size + seg_size,
                      off_min[1] + indexs[j, 1] * seg_size:off_min[1] + indexs[j, 1] * seg_size + seg_size,
                      off_min[2] + indexs[j, 2] * seg_size:off_min[2] + indexs[j, 2] * seg_size + seg_size,
                      ] = batch_outputs.data.cpu()[j - (i - batch_num + 1)].numpy()
        if i%50==0:
            print i," have finished"
    return probs,img_arr




    return output

def doTest(file_name,model):
    probs,img_arr=seg(opt.img_dir+file_name,model)
    probs=probs>opt.prob_threshould 
    probs=morphology.dilation(probs,np.ones([3,3,3]))
    probs=morphology.dilation(probs,np.ones([3,3,3]))
    probs=morphology.erosion(probs,np.ones([3,3,3]))
    labels = measure.label(probs,connectivity=1)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    centers = []
    crops = []
    bboxes = []
    spans=[]
    for prop in regions:
        B = prop.bbox
        if B[3]-B[0]>3 and B[4]-B[1]>5 and B[5]-B[2]>5 :
            z=int((B[3]+B[0])/2.0)
            y=int((B[4]+B[1])/2.0)
            x=int((B[5]+B[2])/2.0)
            span=np.array([int(B[3]-B[0]),int(B[4]-B[1]),int(B[5]-B[2])])
            spans.append(span)
            centers.append(np.array([z,y,x]))
            bboxes.append(B)
    for idx,bbox in enumerate(bboxes):
        crop=np.zeros([opt.crop_size,opt.crop_size,opt.crop_size],dtype=np.float32)
        crop_center=centers[idx]
        half=opt.crop_size/2
        crop_center=check_center(opt.crop_size,crop_center,img_arr.shape)
        crop=img_arr[int(crop_center[0]-half):int(crop_center[0]+half),\
                     int(crop_center[1]-half):int(crop_center[1]+half),\
                     int(crop_center[2]-half):int(crop_center[2]+half)]
        crops.append(crop)
    if opt.is_save:
        np.save(opt.save_dir+file_name+"_nodule.npy",np.array(crops))
        np.save(opt.save_dir+file_name+"_center.npy",np.array(centers))
        np.save(opt.save_dir+file_name+"_size.npy",np.array(spans))
def main(**kwargs):
    parse(kwargs)
    model=Segmentation().cuda().eval()
    model.load(opt.model_dir)
    all_test=glob(opt.img_dir + "*.mhd") 
    print all_test[:10]
    count=0
    for patient in tqdm(all_test):
        file_name=patient.split('/')[-1][:-4]
        doTest(file_name,model)
if __name__ == '__main__':
    fire.Fire()