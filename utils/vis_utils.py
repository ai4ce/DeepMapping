import os
from matplotlib import pyplot as plt
import torch
import numpy as np


def plot_global_point_cloud(point_cloud, pose, valid_points, save_dir, **kwargs):
    if torch.is_tensor(point_cloud):
        point_cloud = point_cloud.cpu().detach().numpy()
    if torch.is_tensor(pose):
        pose = pose.cpu().detach().numpy()
    if torch.is_tensor(valid_points):
        valid_points = valid_points.cpu().detach().numpy()

    file_name = 'global_map_pose'
    if kwargs is not None:
        for k, v in kwargs.items():
            file_name = file_name + '_' + str(k) + '_' + str(v)
    save_name = os.path.join(save_dir, file_name)

    bs = point_cloud.shape[0]
    for i in range(bs):
        current_pc = point_cloud[i, :, :]
        idx = valid_points[i, ] > 0
        current_pc = current_pc[idx]

        plt.plot(current_pc[:, 0], current_pc[:, 1], '.')
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.plot(pose[:, 0], pose[:, 1], color='black')
    plt.savefig(save_name)
    plt.close()

def save_global_point_cloud_open3d(point_cloud,pose,save_dir):
    file_name = 'global_map_pose'
    save_name = os.path.join(save_dir, file_name)

    n_pcd = len(point_cloud)
    for i in range(n_pcd):
        current_pc = np.asarray(point_cloud[i].points)
        plt.plot(current_pc[:, 0], current_pc[:, 1], '.',markersize=1)

    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    plt.plot(pose[:, 0], pose[:, 1], color='black')
    plt.savefig(save_name)
    plt.close()
