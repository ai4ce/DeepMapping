import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)
import torch
import numpy as np

import utils
from dataset_loader import SimulatedPointCloud

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-m','--metric',type=str,default='point',choices=['point','plane'] ,help='minimization metric')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
opt = parser.parse_args()

checkpoint_dir = os.path.join('../results/2D',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

dataset = SimulatedPointCloud(opt.data_dir)
n_pc = len(dataset)

pose_est = np.zeros((n_pc,3),dtype=np.float32)
print('running icp')
for idx in range(n_pc-1):
    dst,valid_dst = dataset[idx] 
    src,valid_src = dataset[idx+1]
    
    dst = dst[valid_dst,:].numpy()
    src = src[valid_src,:].numpy()

    _,R0,t0 = utils.icp(src,dst,metrics=opt.metric)
    if idx == 0: 
        R_cum = R0
        t_cum = t0
    else:
        R_cum = np.matmul(R_cum , R0)
        t_cum = np.matmul(R_cum,t0) + t_cum
    
    pose_est[idx+1,:2] = t_cum.T
    pose_est[idx+1,2] = np.arctan2(R_cum[1,0],R_cum[0,0]) 

save_name = os.path.join(checkpoint_dir,'pose_est.npy')
np.save(save_name,pose_est)

print('saving results')
pose_est = torch.from_numpy(pose_est)
local_pc,valid_id = dataset[:]
global_pc = utils.transform_to_global_2D(pose_est,local_pc)
utils.plot_global_point_cloud(global_pc,pose_est,valid_id,checkpoint_dir)
