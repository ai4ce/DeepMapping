import numpy as np
import open3d
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
