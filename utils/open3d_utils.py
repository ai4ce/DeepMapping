import numpy as np
import open3d as o3d
import copy

def transform_to_global_open3d(pose,local_pcd):
    pcd = copy.deepcopy(local_pcd)
    n_pcd = len(pcd)
    for i in range(n_pcd):
        tx,ty,theta = pose[i,:]
        cos,sin = np.cos(theta),np.sin(theta)
        trans = np.array([
                        [cos,-sin,0,tx],
                        [sin,cos,0,ty],
                        [0,0,1,0],
                        [0,0,0,1],
                        ])
        pcd[i].transform(trans) 
    return pcd


def np_to_pcd(xyz):
    """
    convert numpy array to point cloud object in open3d
    """
    xyz = xyz.reshape(-1,3)
    pcd = o3d.PointCloud()
    pcd.points = o3d.Vector3dVector(xyz)
    pcd.paint_uniform_color(np.random.rand(3,))
    return pcd


def load_obs_global_est(file_name):
    """
    load saved obs_global_est.npy file and convert to point cloud object
    """
    obs_global_est = np.load(file_name)
    n_pc = obs_global_est.shape[0]
    pcds = o3d.PointCloud()

    for i in range(n_pc):
        xyz = obs_global_est[i,:,:]
        current_pcd = np_to_pcd(xyz)
        pcds += current_pcd
    return pcds


