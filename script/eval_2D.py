import set_path
import os
import argparse
import functools
print = functools.partial(print,flush=True)

import numpy as np
import scipy.io as sio

import utils

parser = argparse.ArgumentParser()
parser.add_argument('-c','--checkpoint_dir',type=str,required=True,help='path to results folder')
opt = parser.parse_args()
saved_json_file = os.path.join(opt.checkpoint_dir,'opt.json')
train_opt = utils.load_opt_from_json(saved_json_file)
name = train_opt['name']
data_dir = train_opt['data_dir']

# load ground truth poses
gt_file = os.path.join(data_dir,'gt_pose.mat')
gt_pose = sio.loadmat(gt_file)
gt_pose = gt_pose['pose']
gt_location = gt_pose[:,:2]

# load predicted poses
pred_file = os.path.join(opt.checkpoint_dir,'pose_est.npy')
pred_pose = np.load(pred_file)
pred_location = pred_pose[:,:2] * 512 # denormalization, tbd

# compute absolute trajectory error (ATE)
ate,aligned_location = utils.compute_ate(pred_location,gt_location) 
print('{}, ate: {}'.format(name,ate))
