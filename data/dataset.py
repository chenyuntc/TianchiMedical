#encoding:utf-8
import torch as t
import torch
from torch.utils import data
import numpy as np
import sys
import random
import pandas as pd
from glob import glob
#sys.path.append("../")
from util import augument,check_center,drop_zero,make_mask,crop,zero_normalize
from config import opt
import Queue


class SegDataset(data.Dataset):
    def __init__(self, augument=False,image_size=64,randn=20):
        '''
        @augument:是否数据增强
        @image_size:送进分割网络的立方体的大小
        @randn:切正样本时，以节点为中心，随机偏移的幅度
        ！TODO：训练数据加载，每次返回一个数据立方体块以及相应的mask
        '''
        self.randn=randn
        self.df_node=pd.read_csv(opt.candidate_center)#之前分割网络产生激活的位置中心
        self.df_drop=None#self.df_drop的子集，根据文件名删选出来的
        self.imgfiles = glob(opt.data_train+'*.mhd')#全部训练样本
        self._len = 1244*2#训练集的大小
        self.count=0#训练样本计数器
        self.image_size=image_size
        self.file_num=len(self.imgfiles)#训练样本的数目
        self.file_index=range(len(self.imgfiles))#训练样本的索引
        self.image=None
        self.image_mask=None
        self.nodule_centers=[]
        self.augument=augument
        self.imgdeque=Queue.Queue(maxsize=100)
        self.maskdeque=Queue.Queue(maxsize=100)
    def _select(self,file_name):
        '''
        @file_name:待查找文件名
        Return：根据文件名查找到相应的候选结点，并返回pandas对象
        ！TODO：根据文件名，从self.df_node查找相应的候选结点
        '''
        name=file_name.split('/')[-1][:-4]
        df_drop=self.df_node[self.df_node['seriesuid']==name]
        return df_drop
    def _shuffle(self):
        '''
        Return：None
        ！TODO：1.当所有训练样本被使用一遍，即计数器self.indexx==self.file_num后计数器清零
             2.计数器为0时，将训练样本索引self.file_index随机打乱
        '''
        if self.count==self.file_num:##所有训练样本都使用一次后，计数器归0
            print "计数器清零"
            self.indexx=0
        if self.count==0:#所有训练样本都使用一次后，将索引随机打乱，再次使用
            random.shuffle(self.file_index)
    def _crop_from_center(self,crop_neg=True):
        '''
        @crop_neg:是否切负样本
        Return：返回2个numpy数组
        ！TODO：len(self.nodule_centers)为该样本上结点的数目，记为N
             从原始样本以结点中心切N个正样本
             if crop_neg=True，则以随机中心切N个负样本
        '''
        num=len(self.nodule_centers)
        img=np.zeros([num*2]+3*[self.image_size])
        mask=np.zeros([num*2]+3*[self.image_size])
        for index in range(num):
            offset=np.array([0,0,0])
            if self.randn>0:
                offset=np.random.randint(-1*self.randn,self.randn,3)
            crop_center=self.nodule_centers[index]+offset
            crop_center=check_center(self.image_size,crop_center,self.image.shape)
            cubic_img,cubic_mask=crop(self.image,self.image_mask,v_center=crop_center,width=self.image_size)
            img[index]=normalize(cubic_img)
            mask[index]=cubic_mask
            if crop_neg:#以随机中心切负样本
                cubic_img,cubic_mask=crop(self.image,self.image_mask,width=size)
                if np.sum(cubic_mask)==0:
                    img[index+num]=normalize(cubic_img)
                    mask[index+num]=cubic_mask
        return img,mask
        
    def _crop_from_candidate(self):
        '''
        Return：返回2个numpy数组
        ！TODO：self.df_drop保存了根据文件名选择的候选中心
             len(self.nodule_centers)为该样本上结点的数目，记为N
             从候选中心中选择N个中心，并在原始样本及mask上切出相应的块             
        '''
        num=self.df_drop.shape[0]
        img=np.zeros([num]+3*[self.image_size])
        mask=np.zeros([num]+3*[self.image_size])
        if num>0:
            df_index=np.random.randint(0,num,len(self.nodule_centers))
            for ip in range(len(df_index)):
                tmp=self.df_drop.iloc[df_index[ip]]
                v_center=np.array([tmp.coordZ,tmp.coordY,tmp.coordX])
                v_center=check_center(self.image_size,v_center,self.image.shape)
                cubic_img,cubic_mask=crop(self.image,self.image_mask,v_center=v_center,width=self.image_size)
                img[ip]=normalize(cubic_img)
                mask[ip]=cubic_mask
        return img,mask
    
    def __getitem__(self,index):
        if self.imgdeque.empty():
            self._shuffle()
            ii=self.file_index[self.count]
            self.image,self.image_mask,self.nodule_centers=make_mask(self.imgfiles[ii])
            self.indexx=self.count+1
            self.df_drop=self._select(self.imgfiles[ii])
            img,mask= self.doCrop(crop_neg=False)#从训练样本上切一定数目的立方体以及相应的mask                            
            for i in range(img.shape[0]):#将立方体放入队列
                self.imgdeque.put(img[i])
                self.maskdeque.put(mask[i])
        img=self.imgdeque.get()
        mask=self.maskdeque.get()
        if self.augument:
            img,mask =augument(img,mask)
        
        img_tensor = torch.from_numpy(img.astype(np.float32)).view(1,64,64,64)
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).view(1,64,64,64)
        return img_tensor, mask_tensor
        
    def doCrop(self,crop_neg=True):
        '''
        @file_name:文件名
        @crop_neg:是否切负样本
        ！TODO：从训练样本上切立方体送入分割网络，以样本每个节点为中心切正样本，以随机的中心切负样本
             从候选中心切与正样本相同数目的样本
        '''
        image1,mask1=self._crop_from_candidate()
        image2,mask2=self._crop_from_center(crop_neg)
        image=np.concatenate((image1,image2),0)
        mask=np.concatenate((mask1,mask2),0)
        image,mask=drop_zero(image,mask)
        
        return image,mask    
    def __len__(self):
        return self._len    
    
    
class ClsDataset(data.Dataset):
    '''分类网络的输入'''
    def __init__(self,augument=True,val=False):
        self.augument=augument
        self.ratio=opt.ratio + 1
        pos_cubic=glob(opt.nodule_cubic+"*.npy")
        neg_cubic=glob(opt.candidate_cubic+"*.npy")
        self.val = val        
        if self.val:
            self.pos_cubic = [file for file in pos_cubic if int(file.split('-')[1].split('_')[0])>800]
            self.neg_cubic = [file for file in neg_cubic if int(file.split('-')[1].split('_')[0])>800]
        else :
            self.pos_cubic = [file for file in pos_cubic if int(file.split('-')[1].split('_')[0])<=800]
            self.neg_cubic = [file for file in neg_cubic if int(file.split('-')[1].split('_')[0])<=800]
        



        if self.val: self.ratio = 20 # 如果是验证测试，可以多取一些负样本
        _a = 1 if self.val else 10

        # self.pos_index=range(len(self.pos_cubic))
        self.pos_len = len(self.pos_cubic)        
        self._len=self.ratio*self.pos_len*_a # 希望一个epoch多跑几个

    def __getitem__(self,index):
        
    
        if index%self.ratio==0:
            img=np.load(self.pos_cubic[index/self.ratio%self.pos_len]).astype(np.float32)
            label=1
        else:
            neg_file=self.neg_cubic[np.random.randint(0,len(self.neg_cubic))]
            is_nodule=neg_file.split("/")[-1].split("_")[-1][0]
            img=np.load(neg_file).astype(np.float32)
            label=int(is_nodule)
        z=np.random.randint(-2,2,1)
        yx=np.random.randint(-3,3,2)
        center=np.array([32,32,32])+[z[0],yx[0],yx[1]]
        img=img[center[0]-10:center[0]+10,center[1]-18:center[1]+18,center[2]-18:center[2]+18]
        if self.augument:
            img=augument(img)
        img=zero_normalize(img)
        try:
            img_tensor = torch.from_numpy(img.copy()).view(1,20,36,36)
            # 增加噪声 
            if self.val: noise = 0
            else:  noise = torch.randn(img_tensor.size())*0.2
            img_tensor = img_tensor + noise
        except Exception as e:
            print '!!!!!!!!!!!!!!exception!!!!!!!!!!!!!!!%s' %(str(e))
            return self.__getitem__(index+1)
        return img_tensor, label
 
    def __len__(self):
        return self._len 


def main():
    dataset=ClsDataset()
    jl=0
    dataloader = t.utils.data.DataLoader(dataset,1,       
                num_workers=4,
                shuffle=True,
                pin_memory=True)
    
    for ii, (input, label) in enumerate(dataloader):
        if label[0]==0:
            jl+=1
            
    print jl
if __name__=="__main__":
    main()