import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import open3d as o3d

from dataset_loader import AVD
import utils

def add_y_coord_for_evaluation(pred_pos_DM):
    """
    pred_pos_DM (predicted position) estimated from DeepMapping only has x and z coordinate
    convert this to <x,y=0,z> for evaluation
    """
    n = pred_pos_DM.shape[0]
    x = pred_pos_DM[:,0]
    y = np.zeros_like(x)
    z = pred_pos_DM[:,1]
    return np.stack((x,y,z),axis=-1)

parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint_dir',type=str,required=True,help='path to results folder')
opt = parser.parse_args()
saved_json_file = os.path.join(opt.checkpoint_dir,'opt.json')
train_opt = utils.load_opt_from_json(saved_json_file)
name = train_opt['name']
data_dir = train_opt['data_dir']
subsample_rate = train_opt['subsample_rate']
traj = train_opt['traj']

# load ground truth poses
dataset = AVD(data_dir,traj,subsample_rate)
gt_pose = dataset.gt 
gt_location = gt_pose[:,:3]

# load predicted poses
pred_file = os.path.join(opt.checkpoint_dir,'pose_est.npy')
pred_pose = np.load(pred_file)
pred_location = pred_pose[:,:2] * dataset.depth_scale # denormalization
pred_location = add_y_coord_for_evaluation(pred_location)

# compute absolute trajectory error (ATE)
ate,aligned_location = utils.compute_ate(pred_location,gt_location) 
print('{}, ate: {}'.format(name,ate))

# vis results
global_point_cloud_file = os.path.join(opt.checkpoint_dir,'obs_global_est.npy')
pcds = utils.load_obs_global_est(global_point_cloud_file)
o3d.draw_geometries([pcds])
