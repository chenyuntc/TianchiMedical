import pandas as pd
from glob import  glob
from tqdm import tqdm
import numpy as np
from data.util import load_ct,check_center
from config import opt
def is_exist(lists,f):
    for f_name in lists:
        if f in f_name:
            return True
    return False
        
def crop_nodule():
    '''
    crop roi from the train dataset. for each train img,crop the nodule area in a rectangle
    then reverse it in 3 ways to augment it as the positive samples.
    random choose 10 point as the area center,crop the area as the negative samples
    '''
    df_node = pd.read_csv(opt.annotatiion_csv)
    file_list=glob(opt.data_train+"*.mhd")
    for fcount, img_file in enumerate(tqdm(file_list)):
        file_name=img_file.split('/')[-1][:-4]
        print "doing on ",file_name
        mini_df = df_node[df_node["seriesuid"]==file_name] #get all nodules associate with file
        if mini_df.shape[0]>0: 
            img_arr,origin,spacing=load_ct(img_file)           
            patient_nodules=[]
            for node_idx, cur_row in mini_df.iterrows():   
                crop_img=np.zeros([64,64,64])
                diam = cur_row["diameter_mm"]
                center = np.array([cur_row["coordZ"], cur_row["coordY"], cur_row["coordX"]])   # nodule center
                v_center = np.rint((center-origin)/spacing)  # nodule center in voxel space (still x,y,z ordering)
                v_center = check_center(64,v_center,img_arr.shape)
                v_center=v_center.astype(np.int32)
                crop_img=img_arr[v_center[0]-32:v_center[0]+32,v_center[1]-32:v_center[1]+32,v_center[2]-32:v_center[2]+32]
                
                np.save(opt.nodule_cubic+file_name+"_"+str(node_idx)+".npy", crop_img)
def crop_candidate():
    df_node = pd.read_csv(opt.candidate_center)
    file_list=glob(opt.data_train+"*.mhd")
    done=glob(opt.candidate_cubic+'*.npy')
    for fcount, img_file in enumerate(tqdm(file_list)):
        file_name=img_file.split('/')[-1][:-4]
        if is_exist(done,file_name):
            continue
        print "doing on ",file_name
        mini_df = df_node[df_node["seriesuid"]==file_name] #get all nodules associate with file
        if mini_df.shape[0]>0: 
            img_arr,origin,spacing=load_ct(img_file)           
            patient_nodules=[]
            for node_idx, cur_row in mini_df.iterrows():   
                crop_img=np.zeros([64,64,64])
                is_nodule=cur_row["isnodule"]
                center = np.array([cur_row["coordZ"], cur_row["coordY"], cur_row["coordX"]])   # nodule center
                v_center = check_center(64,center,img_arr.shape)
                v_center=v_center.astype(np.int32)
                crop_img=img_arr[v_center[0]-32:v_center[0]+32,v_center[1]-32:v_center[1]+32,v_center[2]-32:v_center[2]+32]
                
                np.save(opt.candidate_cubic+file_name+"_"+str(node_idx)+"_"+str(int(is_nodule))+".npy", crop_img)
        
        
crop_candidate()    
#crop_nodule() 