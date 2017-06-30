#encoding:utf-8
import numpy as np
import SimpleITK as sitk
from config import opt
import pandas as pd
def get_file(lists,fname):
    for f in lists:
        if fname in f:
            return f
def vote(arr):
    pos_prob=arr[:,1]
    votes=0
    return np.mean(pos_prob)
def get_topn(arr,n):
    tmp=np.sort(arr)
    index=[]
    for i in range(n):
        kk=np.where(arr==tmp[-1-i])[0]
        index.append(kk[0])
    return index    
def rotate(imgs,type):
    '''
    @imgs:带翻转的图像
    @type：以何种方式翻转
    Return：翻转后的图像
    ！TODO：对图像以某种方式进行翻转
    '''
    if type==0:
        return imgs[::-1,:,:]
    elif type==1:
        return imgs[:,::-1,:]
    elif type==2:
        return imgs[:,:,::-1]
    elif type==3:
        return imgs[:,::-1,::-1]
    elif type==4:
        return imgs[::-1,:,::-1]
    elif type==5:
        return imgs[::-1,::-1,:]
    elif type==6:
        return imgs[::-1,::-1,::-1]
    else:
        return imgs
    
def augument(imgs,mask=None):
    '''
    @imgs：原始待增强数据
    @mask：原始数据对应的mask，数据分类时，没有mask
    ！TODO：数据增强
    '''
    type=np.random.randint(0,7)
    if mask is None:
        return rotate(imgs,type)
    else:
        return rotate(imgs,type),rotate(mask,type)
def zero_normalize(image, mean=-600.0, var=-300.0):
        image = (image - mean) / (var)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
def normalize(image, MIN_BOUND=-1000.0, MAX_BOUND=400.0):
    '''
    @image：原始数据
    @MIN_BOUND：最小值
    @MAX_BOUND：最大值
    Return：归一化后的image
    ！TODO：数据截断归一化
    '''
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image    
def crop(image,image_mask,v_center=None,width=64):
    '''
    @image：原始图像数据
    @image_mask：图像对应的二值mask
    @v_center：剪切中心，if None，则随机产生中心
    @width：剪切的大小
    Return：从image和image_mask上以v_center为中心剪切下来的边长为width的立方体
    ！TODO：数据截断归一化
    '''
    half=width/2
    cubic_img=np.zeros([width,width,width],dtype=np.float32)
    cubic_mask=np.zeros([width,width,width],dtype=np.float32)
    if v_center is None:
        z,x,y=image.shape
        center_z=np.random.randint(half+1,z-half-1,1)
        center_xy=np.random.randint(half+1,x-half-1,2)
        v_center=np.concatenate([center_z,center_xy])
    cubic_img[:,:,:]=image[int(v_center[0]-half):int(v_center[0]+half),\
                int(v_center[1]-half):int(v_center[1]+half),\
                int(v_center[2]-half):int(v_center[2]+half)]
    cubic_mask[:,:,:]=image_mask[int(v_center[0]-half):int(v_center[0]+half),\
                int(v_center[1]-half):int(v_center[1]+half),\
                int(v_center[2]-half):int(v_center[2]+half)]
    return cubic_img,cubic_mask    
def drop_zero(image,mask):
    '''
    @image：Numpy【N,D,D,D】，待检查图像，去除image【i】为0的元素
    @mask：与mask同大小的Numpy，去除image【i】为0处的元素
    Return：去0后的image和mask
    ！TODO：1.找到image[i]全为0的位置index
         2.把该位置处的image【i】，mask【i】删掉
    '''
    sum_=np.sum(image,axis=(1,2,3))
    index=sum_!=0
    return image[index],mask[index]
    
def check_center(size,crop_center,image_shape):
    '''
    @size：所切块的大小
    @crop_center：待检查的切块中心
    @image_shape：原图大小
    Return：检查修正后切块中心
    ！TODO：在一张图上以crop_center为中心切一个size大小块，检查所切块是否会超出图像范围，返回合适的切块中心
    '''
    half=size/2
    margin_min=crop_center-half#检查下界
    margin_max=crop_center+half-image_shape#检查上界
    for i in range(3):#如有超出，对中心进行修正
        if margin_min[i]<0:
            crop_center[i]=crop_center[i]-margin_min[i]
        if margin_max[i]>0:
            crop_center[i]=crop_center[i]-margin_max[i]
    return crop_center
def load_ct(file_name):
    mhd=sitk.ReadImage(file_name)
    img_arr=sitk.GetArrayFromImage(mhd)
    origin = np.array(mhd.GetOrigin())
    spacing = np.array(mhd.GetSpacing())
    return img_arr,origin[::-1],spacing[::-1]

def make_mask(file_name):
    '''
    @file_name:待计算mask的样本
    Return：样本的mask
    ！TODO：为训练样本制作二值mask
    '''
    image,origin,spacing=load_ct(file_name)
    df_node= pd.read_csv(opt.annotatiion_csv)
    name=file_name.split('/')[-1][:-4]
    df_node = df_node[df_node['seriesuid']==name]
    image_mask=np.zeros(image.shape,dtype=np.float32)
    nodule_centers=[]
    for index, nodule in df_node.iterrows():
        if True:
            coord_z,coord_y,coord_x=nodule.coordZ, nodule.coordY, nodule.coordX
            nodule_center = np.array([coord_z,coord_y,coord_x])
            v_center = np.rint((nodule_center - origin) / spacing)
            v_center = np.array(v_center, dtype=int)
            nodule_centers.append(v_center)
            radius=nodule.diameter_mm/2
            span=np.round(radius/spacing)
            image_mask[np.clip(int(v_center[0]-span[0]),0,image.shape[0]):np.clip(int(v_center[0]+span[0]+1),0,image.shape[0]),\
                   np.clip(int(v_center[1]-span[1]),0,image.shape[1]):np.clip(int(v_center[1]+span[1]+1),0,image.shape[1]),\
                   np.clip(int(v_center[2]-span[2]),0,image.shape[2]):np.clip(int(v_center[2]+span[2]+1),0,image.shape[2])]=int(1)
   
    return image,image_mask,nodule_centers
