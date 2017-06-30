from glob import glob 
from tqdm import tqdm
import csv
import numpy as np
import pandas as pd
from config import opt
from common_del.cysb import select,voxel_2_world
center_list=glob('/mnt/7/0629_train_no_normalization/*_center.npy')
num_nodule=0
flag=0
f=open(opt.candidate_center, "wa")
csv_writer = csv.writer(f, dialect="excel")
csv_writer.writerow(['seriesuid', 'coordX', 'coordY', 'coordZ','isnodule'])
for file in tqdm(center_list):
    file_name=file.split('/')[-1].split("_")[0]
    real_center=select(file_name,"train")
    aa=np.load(file)
    center_now_world=voxel_2_world(aa[:,::-1],file_name)
    span=voxel_2_world([48,48,48],file_name)/2
    length=aa.shape[0]
    print "nodule nums: ",length
    cysb=0
    for j in range(0,length):
        flag=0
        center_now=center_now_world[j]
        for i_,nodule in real_center.iterrows():   
            center=np.array([nodule['coordX'],nodule['coordY'],nodule['coordZ']])
            radius=nodule['diameter_mm']/2
            diff=np.abs(center-center_now)
            distance=np.sqrt((diff**2).sum())
            if distance<radius:
                print "I,m a real nodule"
                num_nodule=num_nodule+1
                row=[file_name]+list(aa[j,::-1])+[1]
                csv_writer.writerow(row)
                continue
            elif np.any(diff>span+radius+2):
                flag=0
            else:
                flag=-1
                cysb+=1
                continue
        if flag==0:
            row=[file_name]+list(aa[j,::-1])+[0]
            csv_writer.writerow(row)
    print "I know Chen Yun Sha Bi,drop it,cysb %d times"%cysb
    #break
print num_nodule
            

                
