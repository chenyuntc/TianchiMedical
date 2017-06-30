#encoding:utf-8
import os
import sys
import time
from glob import glob

import numpy as np
from skimage import color, data, measure, morphology, segmentation

import torch as t
from common.cysb import cropBlocks, get_ct, normalize, resample
from common.util import Visualizer, get_optimizer
from Segmentation import Segmentation
from torch.utils.data import DataLoader

sys.path.append('../')
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x


def seg(file_name,model):
        seg_size = 64
        print " data prepared................................."
        img_arr,origin,spacing=get_ct(self.img_dir+file_name+'.mhd')
        #img_arr,spacing=resample(img_arr,spacing)
        img_new=normalize(img_arr)
        depth, height, width = img_new.shape
        blocks, indexs = cropBlocks(img_new)
        probs = np.zeros(img_new.shape, dtype=np.float32)
        num = np.array(img_new.shape) / 64
        off = np.array(img_new.shape) - 64 * num
        off_min = off / 2
        batch_num=4
        print "doing on patient:", file_name

        for i in range(blocks.shape[0]):
            if (i % batch_num == batch_num - 1):
                batch_inputs_numpy = [torch.from_numpy(blocks[j][np.newaxis, np.newaxis, :, :, :]) for j in range(i - batch_num + 1, i + 1)]
                batch_inputs = torch.autograd.Variable(torch.cat(batch_inputs_numpy, 0), volatile=True).cuda()
                batch_outputs = model(batch_inputs)
                for j in range(i - batch_num + 1, i + 1):
                    probs[off_min[0] + indexs[j, 0] * 64:off_min[0] + indexs[j, 0] * 64 + 64,
                          off_min[1] + indexs[j, 1] * 64:off_min[1] + indexs[j, 1] * 64 + 64,
                          off_min[2] + indexs[j, 2] * 64:off_min[2] + indexs[j, 2] * 64 + 64,
                          ] = batch_outputs.data.cpu()[j - (i - batch_num + 1)].numpy()
            if i%50==0:
                print i," have finished
        return probs




    return output
class Tester(object):
    def __init__(self,img_dir,save_dir,model_dir,is_save=True):
        self.img_dir=img_dir
        self.save_dir=save_dir
        self.model_dir=model_dir
        self.is_save=is_save
        self.model=Segmentation().cuda().eval()
        self.model.load(self.model_dir)
    def doTest(self,file_name):
        seg(self.img_dir+file_name,self.model)
        probs=probs>0.8  
        probs=morphology.dilation(probs,np.ones([3,3,3]))
        probs=morphology.dilation(probs,np.ones([3,3,3]))
        probs=morphology.erosion(probs,np.ones([3,3,3]))
        labels = measure.label(probs,connectivity=1)
        label_vals = np.unique(labels)
        regions = measure.regionprops(labels)
        centers = []
        crops = []
        bboxes = []
        for prop in regions:
            B = prop.bbox
            if B[3]-B[0]>4 and B[4]-B[1]>4 and B[5]-B[2]>4 :
                z=int((B[3]+B[0])/2.0)
                y=int((B[4]+B[1])/2.0)
                x=int((B[5]+B[2])/2.0)
                span=np.array([int((B[3]-B[0])/2.0),int((B[4]-B[1])/2.0),int((B[5]-B[2])/2.0)])
                centers.append(np.array([z,y,x]))
                bboxes.append(B)
        for idx,bbox in enumerate(bboxes):
            crop=np.zeros([48,48,48],dtype=np.float32)
            crop_center=centers[idx]
            min_margin=crop_center-24
            max_margin=crop_center+24-np.array(img_new.shape)
            for i in range(3):
                if min_margin[i]<0:
                    crop_center[i]=crop_center[i]-min_margin[i]
                if max_margin[i]>0:
                    crop_center[i]=crop_center[i]-max_margin[i]
            crop=img_new[int(crop_center[0]-24):int(crop_center[0]+24),\
                         int(crop_center[1]-24):int(crop_center[1]+24),\
                         int(crop_center[2]-24):int(crop_center[2]+24)]
            crops.append(crop)
        if self.is_save:
            np.save(self.save_dir+file_name+"_nodule.npy",np.array(crops))
            np.save(self.save_dir+file_name+"_center.npy",np.array(centers))
if __name__ == '__main__':
    img_dir='/home/x/data/datasets/tianchi/train/'
    save_dir='/mnt/7/nodule_test_without/nodule_train_dilation2_erosion1/'
    model_dir="../../checkpoint/seg_0527_05:10:49.pth"
    #model_dir="checkpoints/seg_0622_14:06:32.pth"
    test=Tester(img_dir,save_dir,model_dir)
    all_test=glob(img_dir + "*.mhd") 
    print all_test[:10]
    for patient in tqdm(all_test):
        file_name=patient.split('/')[-1][:-4]
        test.doTest(file_name)    
