#coding:utf8
import pandas as pd
import random
import numpy as np
import csv
from tqdm import tqdm
def pcsv(file_name,out_file=None):
    '''
    处理csv文件：
    - 把每个文件的最大的概率加大
    - 座标shuffle
    shuffle座标
    python process.py csv input.csv result.csv
    '''
    f = pd.read_csv(file_name)
    ids = set(f.seriesuid.tolist())
    sum_=0
    for id in ids:
        # ct文件中概率最大的结节的概率再给它大点，0.6->0.96
      
        max_index = f[f.seriesuid==id].probability.argmax()
        probability = f.iloc[max_index,4]
        if len(f.iloc[max_index])>5:sum_+=f.iloc[max_index,5]
        if probability>0.2:   f.iloc[max_index,4] = probability*0.05+0.95
        # probability = f.iloc[max_index,4]
        
        
        # probability2 = f.iloc[max_index+1,4]
        # if probability- probability2<0.1:f.iloc[max_index+1,4]=0.*probability2+0.8#f.iloc[max_index,4]*0.8
    #     # if probability-probability2>0.4:f.iloc[max_index+1,4]=probability2**2
        
    #     # if probability>0.8: f.iloc[max_index+1,4] = probability*0.6+0.4
    #     # f.iloc[max_index+1,4] = f.iloc[max_index + 2,4]*0.7 + 0.3*f.iloc[max_index+3,4]
        
    #     # probability = f.iloc[max_index+2,4]
    #     # if probability>0.7: f.iloc[max_index+2,4] = probability*0.6+0.4

    for ii in range(len(f)):
        # shuffle 座标
        record = f.iloc[ii]
        coords = record.coordX,record.coordY,record.coordZ
        x,y,z = [_+0.4*random.random() for _  in coords]
        f.iloc[ii,1],f.iloc[ii,2],f.iloc[ii,3]=x,y,z

    if out_file is None:
        out_file = file_name
    print sum_
    f.to_csv(out_file,index=False)

def check_nodule(file_name,out_file):
    '''
    1. 判断文件中的节点是不是正样本还是负样本
    2. 如果是正样本，写上半径，否则写上-1
    !TODO: 如果是半径是否要写上seg找出来的半径, modify Seg.py
    '''
    #import ipdb;ipdb.set_trace()
    from data.util import select
    f = pd.read_csv(file_name)
    out_file=open(out_file,"wa")
    csv_writer = csv.writer(out_file, dialect="excel")
    csv_writer.writerow(
        ['seriesuid', 'coordX', 'coordY', 'coordZ', 'probability','isnodule','diameter_mm'])
    ids = set(f.seriesuid.tolist())
    print "file_nums: ",len(ids)
    for seriesuid in tqdm(ids):
        # if seriesuid=='LKDS-00826':import ipdb;ipdb.set_trace()
        real_center=select(seriesuid)
        center_now_world=f[f.seriesuid==seriesuid]
        for i,nodule in center_now_world.iterrows():
            center_now=np.array([nodule['coordX'],nodule['coordY'],nodule['coordZ']])
            probability=nodule['probability']
            row=[seriesuid]+list(center_now)+[nodule['probability']]
            flag=0
            for j_,nodule_ in real_center.iterrows():
                real=np.array([nodule_['coordX'],nodule_['coordY'],nodule_['coordZ']])
                radius=nodule_['diameter_mm']/2
                diff=np.abs(real-center_now)
                distance=np.sqrt((diff**2).sum())
                if distance<radius: 
                    flag=1
                    row=row+[1,nodule_['diameter_mm']]
                    break
       
            if flag==0:
                if len(row)>5:import ipdb;ipdb.set_trace()
                row=row+[0,-1.]
            if len(row)>7:row=row[:-2]#import ipdb;ipdb.set_trace()
            csv_writer.writerow(row)

                
                    
                
        
    
        


if __name__=='__main__':
    #check_nodule("/home/x/dcsb/refactor/train_no.csv")
    import fire
    fire.Fire()
