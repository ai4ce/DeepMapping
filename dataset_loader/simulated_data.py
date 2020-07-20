import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from open3d import read_point_cloud

import utils


def find_valid_points(local_point_cloud):
    """
    find valid points in local point cloud
        invalid points have all zeros local coordinates
    local_point_cloud: <BxNxk> 
    valid_points: <BxN> indices  of valid point (0/1)
    """
    eps = 1e-6
    non_zero_coord = torch.abs(local_point_cloud) > eps
    valid_points = torch.sum(non_zero_coord, dim=-1)
    valid_points = valid_points > 0
    return valid_points


class SimulatedPointCloud(Dataset):
    def __init__(self, root, trans_by_pose=None):
        # trans_by_pose: <Bx3> pose
        self.root = os.path.expanduser(root)
        self._trans_by_pose = trans_by_pose
        file_list = glob.glob(os.path.join(self.root, '*pcd'))
        self.file_list = sorted(file_list)

        self.pcds = [] # a list of open3d pcd objects 
        point_clouds = [] #a list of tensor <Lx2>
        for file in self.file_list:
            pcd = read_point_cloud(file)
            self.pcds.append(pcd)

            current_point_cloud = np.asarray(pcd.points, dtype=np.float32)[:, 0:2]
            point_clouds.append(current_point_cloud)

        point_clouds = np.asarray(point_clouds)
        self.point_clouds = torch.from_numpy(point_clouds) # <NxLx2>

        self.valid_points = find_valid_points(self.point_clouds) # <NxL>

        # number of points in each point cloud
        self.n_obs = self.point_clouds.shape[1]

    def __getitem__(self, index):
        pcd = self.point_clouds[index,:,:]  # <Lx2>
        valid_points = self.valid_points[index,:]
        if self._trans_by_pose is not None:
            pcd = pcd.unsqueeze(0)  # <1XLx2>
            pose = self._trans_by_pose[index, :].unsqueeze(0)  # <1x3>
            pcd = utils.transform_to_global_2D(pose, pcd).squeeze(0)
        else:
            pose = torch.zeros(1,3,dtype=torch.float32)
        return pcd,valid_points,pose

    def __len__(self):
        return len(self.point_clouds)
