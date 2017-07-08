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
from util import augument,check_center,drop_zero,make_mask,crop,zero_normalize,normalize
from config import opt
import Queue

class SegDataLoader(data.Dataset):
    def __init__(self,augument=True,image_size=64,randn=7,val=False):
        '''
        @augument:是否数据增强
        @image_size:送进分割网络的立方体的大小
        @randn:切正样本时，以节点为中心，随机偏移的幅度
        ！TODO：训练数据加载，每次返回一个数据立方体块以及相应的mask
        '''
        self.randn=randn
        self.augument=augument
        self.df_node=pd.read_csv(opt.annotatiion_csv)#标记的结点信息
        self.df_candiadate=pd.read_csv(opt.candidate_center)##分割网络产生的疑似结点中心位置保存路径
        self.imgfiles = glob(opt.data_train+'*.mhd')#全部训练样本
        self.pos_cubic=glob(opt.nodule_cubic+"*.npy")
        self.neg_cubic=glob('/mnt/7/0701_train_nodule_candidate/'+"*npz")
        self._len = 1244*opt.seg_ratio
    def _crop(self,mask,voxel_center):
        '''
        从image，mask上剪切48大小的块，送入分割网络
        @image：numpy（64,64,64）待剪切的数据图像，从上剪切48大小
        @mask：待剪切的mask，从上剪切48大小
        voxel_center：mask上剪切的中心
        Return
            crop_img，numpy（48,48,48）图像数据
            crop_mask，numpy（48,48,48）图像数据对应的mask
        '''
        v_center = check_center(64,voxel_center,mask.shape)#检查是否剪切超出图像范围
        v_center=v_center.astype(np.int32)
        half=32
        #剪切mask
        crop_mask=mask[v_center[0]-half:v_center[0]+half,v_center[1]-half:v_center[1]+half,v_center[2]-half:v_center[2]+half]
        return crop_mask
    def _crop_pos(self,index):
        '''
        剪切正样本，从保存的结点文件（64,64,64），剪切（48，48,48）的块
        pos_file:/mnt/t/train_nodule_cubic/LKDS-00354_1021.npy,对应于一个结点
        patient_id:LKDS-00354,为该结点对应的原始图像
        nodule_index:1021 该结点在self.df_node中的index，便于取出该结点的中心
        '''
        file_index=int(index/opt.seg_ratio)
        pos_file=self.pos_cubic[file_index]
        nodule_index=int(pos_file.split('/')[-1].split('_')[1].split('.')[0])
        df_min=self.df_node.iloc[nodule_index]#结点信息，包含他的中心世界坐标，文件名
        patient_id=df_min.seriesuid
        world_center=np.array([df_min.coordZ,df_min.coordY,df_min.coordX],dtype=np.float32)
        mhd_file=opt.data_train+patient_id+'.mhd'#结点对应训练集的mhd文件
        image_mask,nodule_centers,origin,spacing=make_mask(mhd_file)#生成mask，并返回其原图的节点信息
        img=np.load(pos_file).astype(np.float32)
        voxel_center= np.rint((world_center-origin)/spacing)
        crop_mask=self._crop(image_mask,voxel_center)
        return img,crop_mask
    def _crop_neg(self,index):
        N=len(self.neg_cubic)#疑似位置总数目
        ids=np.random.randint(0,N)
        neg_file=self.neg_cubic[ids]
        data=np.load(neg_file)#疑似结点数据文件，字典，包含图像数据以及对应原图上的体素坐标中心
        voxel_center=data['center']
        patient_id=neg_file.split('/')[-1].split("_")[0]
        mhd_file=opt.data_train+patient_id+'.mhd'#结点对应训练集的mhd文件
        image_mask,nodule_centers,origin,spacing=make_mask(mhd_file)#生成mask，并返回其原图的节点信息
        img=data['data'].astype(np.float32)
        crop_mask=self._crop(image_mask,voxel_center)
        return img,crop_mask 
        
        
    def __getitem__(self,index):
        if index%opt.seg_ratio==0:#取正样本
            img,mask=self._crop_pos(index)
        else:
            img,mask=self._crop_neg(index)
        offset=np.array([0,0,0])
        if self.randn>0:
            offset=np.random.randint(-1*self.randn,self.randn,3)
        center=[32,32,32]+offset
        half=opt.train_crop_size/2
        mask=mask[(center[0]-half):(center[0]+half),(center[1]-half):(center[1]+half),(center[2]-half):(center[2]+half)]
        img=img[(center[0]-half):(center[0]+half),(center[1]-half):(center[1]+half),(center[2]-half):(center[2]+half)]
        if self.augument:
            img,mask=augument(img,mask)
        scale=48
        try:
            img_tensor = torch.from_numpy(img.astype(np.float32)).view(1,scale,scale,scale)
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).view(1,scale,scale,scale)
            return img_tensor, mask_tensor 
        except:
            print 'er'
            return self.__getitem__(index+1)
          
            
    def __len__(self):
        return self._len 
            
class ClsDataset2(data.Dataset):
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
        img=img[center[0]-20:center[0]+20,center[1]-20:center[1]+20,center[2]-20:center[2]+20]
        if self.augument:
            img=augument(img)
        img=zero_normalize(img)
        try:
            img_tensor = torch.from_numpy(img.copy()).view(1,40,40,40)
            # 增加噪声 
            if self.val: noise = 0
            # 原来的噪声标准差是0.2. 数据集中标准差大概是0.4~0.5
            else:  noise = torch.randn(img_tensor.size())*0.05
            img_tensor = img_tensor + noise
        except Exception as e:
            print '!!!!!!!!!!!!!!exception!!!!!!!!!!!!!!!%s' %(str(e))
            return self.__getitem__(index+1)
        return img_tensor, label
 
    def __len__(self):
        return self._len 

class ClsDataset3(data.Dataset):
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
        _a = 1 if self.val else 5

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
        #z=np.random.randint(-3,3,1)
        zyx=np.random.randint(-3,3,3)
        center=np.array([32,32,32])+[zyx[0],zyx[1],zyx[2]]
        img=img[center[0]-20:center[0]+20,center[1]-20:center[1]+20,center[2]-20:center[2]+20]
        if self.augument:
            img=augument(img)
        img=normalize(img)
        try:
            img_tensor = torch.from_numpy(img.copy()).view(1,40,40,40)
            # 增加噪声 
            if self.val: noise = 0
            # 原来的噪声标准差是0.2. 数据集中标准差大概是0.4~0.5
            else:  noise = torch.randn(img_tensor.size())*0.1
            img_tensor = img_tensor + noise
        except Exception as e:
            print '!!!!!!!!!!!!!!exception!!!!!!!!!!!!!!!%s' %(str(e))
            return self.__getitem__(index+1)
        return img_tensor, label
 
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
            self.pos_cubic = [file for file in pos_cubic if int(file.split('-')[1].split('_')[0])<=1000]
            self.neg_cubic = [file for file in neg_cubic if int(file.split('-')[1].split('_')[0])<=1000]
        



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
        #z=np.random.randint(-3,3,1)
        zyx=np.random.randint(-3,3,3)
        center=np.array([32,32,32])+[zyx[0],zyx[1],zyx[2]]
        img=img[center[0]-10:center[0]+10,center[1]-18:center[1]+18,center[2]-18:center[2]+18]
        if self.augument:
            img=augument(img)
        img=normalize(img)
        try:
            img_tensor = torch.from_numpy(img.copy()).view(1,20,36,36)
            # 增加噪声 
            if self.val: noise = 0
            # 原来的噪声标准差是0.2. 数据集中标准差大概是0.4~0.5
            else:  noise = torch.randn(img_tensor.size())*0.01
            img_tensor = img_tensor + noise
        except Exception as e:
            print '!!!!!!!!!!!!!!exception!!!!!!!!!!!!!!!%s' %(str(e))
            return self.__getitem__(index+1)
        return img_tensor, label
 
    def __len__(self):
        return self._len 


def main():
    dataset=SegDataLoader(randn=7)
    jl=0
    dataloader = t.utils.data.DataLoader(dataset,1,       
                num_workers=4,
                shuffle=True,
                pin_memory=True)
    #start=np.random.randint(0,500)
    for ii, (input, mask) in enumerate(dataloader):
        np.save("del/images_"+str(ii)+".npy",input.numpy())
        np.save("del/masks_"+str(ii)+".npy",mask.numpy())
        if ii==10:
            break
            
if __name__=="__main__":
    main()
