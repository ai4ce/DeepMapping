import os
import glob
import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset
import open3d as o3d

import utils

def find_valid_points(local_point_cloud):
    """
    find valid points in local point cloud
        the z-values for invalid points are zeros
    local_point_cloud: <BxHxWx3> 
    valid_points: <BxHxW> indices of valid point (0/1)
    """
    eps = 1e-6
    valid_points = local_point_cloud[:,:,:,-1] > eps
    return valid_points

class AVD(Dataset):
    width,height = 1080, 1920
    focal_len=(1070,1069.126)
    principal_pt = (927.268, 545.76)

    def __init__(self,root,traj,subsample_rate=20,depth_scale=2000,trans_by_pose=None):
        self.root = root
        self.traj = traj
        self._trans_by_pose = trans_by_pose
        self.depth_scale = depth_scale
        traj_file = os.path.join(root,'local_point_cloud',traj)
        depth_files = [line.rstrip('\n') for line in open(traj_file)]
        self.n_pc = len(depth_files)

        # load point cloud and gt
        image_structs = sio.loadmat(os.path.join(root,'image_structs.mat'))
        image_structs = image_structs['image_structs'][0]
        image_files = [i[0][0] for i in image_structs] 

        point_clouds = []
        self.gt = np.zeros((self.n_pc,6))
        for index,depth_file in enumerate(depth_files):
            depth_file_full = os.path.join(root,'high_res_depth',depth_file)
            depth_map = np.asarray(o3d.read_image(depth_file_full))
            current_point_cloud = utils.convert_depth_map_to_pc(depth_map,self.focal_len,self.principal_pt,depth_scale=self.depth_scale)
            current_point_cloud = current_point_cloud[::subsample_rate,::subsample_rate,:]
            point_clouds.append(current_point_cloud)
            
            image_file = depth_file[:14] + '1.jpg'
            idx = image_files.index(image_file)
            current_image_struct = image_structs[idx]
            current_pos = current_image_struct[6]
            current_direction = current_image_struct[4]
            current_gt = np.concatenate((current_pos,current_direction)).T
            
            self.gt[index,:] = current_gt

        point_clouds = np.asarray(point_clouds)
        self.point_clouds = torch.from_numpy(point_clouds) # <NxHxWx3>
        self.valid_points = find_valid_points(self.point_clouds) 
        
        
    def __getitem__(self,index):
        pcd = self.point_clouds[index,:,:,:]  # <HxWx3>
        valid_points = self.valid_points[index,:]  #<HxW>
        if self._trans_by_pose is not None:
            pcd = pcd.unsqueeze(0)  # <1XHxWx3>
            pose = self._trans_by_pose[index, :].unsqueeze(0)  # <1x3>
            pcd = utils.transform_to_global_AVD(pose, pcd).squeeze(0)
        return pcd,valid_points

    def __len__(self):
        return self.n_pc
