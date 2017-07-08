#encoding:utf-8
import SimpleITK as sitk
import csv
import numpy as np
from glob import glob
import pandas as pd
try:
    from tqdm import tqdm # long waits are not fun
except:
    print('tqdm 是一个轻量级的进度条小包。。。')
    tqdm = lambda x : x
#exit=pd.read_csv("/home/x/dcsb/data/TianChi/csv/information.csv")
#dones=exit['seriesuid']
#dones=dones.values
f=open("/home/x/dcsb3/data/TianChi/csv/information1.csv", "wa")
train_list=glob('/home/x/data/datasets/tianchi/train/*.mhd')
for file_train in (tqdm(train_list)):
    file=file_train.split('/')[-1][:-4]
    mhd=sitk.ReadImage(file_train)
    origin = np.array(mhd.GetOrigin())
    spacing = np.array(mhd.GetSpacing())
    size=np.array(mhd.GetSize())
    row=[file]+list(origin)+list(spacing)+list(size)
    csv_writer = csv.writer(f, dialect = "excel")
    csv_writer.writerow(row)
    print row