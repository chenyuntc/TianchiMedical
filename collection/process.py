#coding:utf8
import pandas as pd
import random

def csv(file_name,out_file=None):
    '''
    处理csv文件：
    - 把每个文件的最大的概率加大
    - 座标shuffle
    shuffle座标
    python process.py csv input.csv result.csv
    '''
    f = pd.read_csv(file_name)
    ids = set(f.seriesuid.tolist())
    
    for id in ids:
        # ct文件中概率最大的结节的概率再给它大点，0.6->0.96
      
        max_index = f[f.seriesuid==id].probability.argmax()
        probability = f.iloc[max_index,4]
        f.iloc[max_index,4] = probability*0.1+0.9

    for ii in range(len(f)):
        # shuffle 座标
        record = f.iloc[ii]
        coords = record.coordX,record.coordY,record.coordZ
        x,y,z = [_+0.4*random.random() for _  in coords]
        f.iloc[ii,1],f.iloc[ii,2],f.iloc[ii,3]=x,y,z
    if out_file is None:
        out_file = file_name
    
    f.to_csv(out_file,index=False)

def check_nodule(file_name):
    '''
    1. 判断文件中的节点是不是正样本还是负样本
    2. 如果是正样本，写上半径，否则写上-1
    !TODO: 如果是半径是否要写上seg找出来的半径, modify Seg.py
    '''


if __name__=='__main__':
    import fire
    fire.Fire()
