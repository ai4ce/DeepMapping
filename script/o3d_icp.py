import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)
import torch
import numpy as np
import open3d

import utils
from dataset_loader import SimulatedPointCloud

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str,default='test',help='experiment name')
parser.add_argument('-m','--metric',type=str,default='point',choices=['point','plane'] ,help='minimization metric')
parser.add_argument('-d','--data_dir',type=str,default='../data/2D/',help='dataset path')
parser.add_argument('-r','--radius',type=float,default=0.02)
opt = parser.parse_args()
print(opt.radius)

checkpoint_dir = os.path.join('../results/2D',opt.name)
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
utils.save_opt(checkpoint_dir,opt)

dataset = SimulatedPointCloud(opt.data_dir)
local_pcds = dataset.pcds[:]
n_pc = len(local_pcds)

#"""
# remove invalid points
local_pcds = [utils.remove_invalid_pcd(x) for x in local_pcds]

if opt.metric == 'point':
    metric = open3d.TransformationEstimationPointToPoint() 
else:
    metric = open3d.TransformationEstimationPointToPlane() 
    for idx in range(n_pc):
        open3d.estimate_normals(local_pcds[idx],search_param = open3d.KDTreeSearchParamHybrid(radius=opt.radius,max_nn=10))


pose_est = np.zeros((n_pc,3),dtype=np.float32)
print('running icp')
for idx in range(n_pc-1):
    dst = local_pcds[idx]
    src = local_pcds[idx+1]
    result_icp = open3d.registration_icp(src,dst,opt.radius,estimation_method=metric)

    R0 = result_icp.transformation[:2,:2]
    t0 = result_icp.transformation[:2,3:]
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

# plot point cloud in global frame

print('saving results')
global_pcds = utils.transform_to_global_open3d(pose_est,local_pcds)
utils.save_global_point_cloud_open3d(global_pcds,pose_est,checkpoint_dir)
