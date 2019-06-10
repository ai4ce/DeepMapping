from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from .networks import LocNetReg, MLP
from utils import transform_to_global_2D


def get_M_net_inputs_labels(occupied_points, unoccupited_points):
    n_pos = occupied_points.shape[1]
    inputs = torch.cat((occupied_points, unoccupited_points), 1)
    bs, N, _ = inputs.shape

    gt = torch.zeros([bs, N, 1], device=occupied_points.device)
    gt.requires_grad_(False)
    gt[:, :n_pos, :] = 1
    return inputs, gt


def sample_unoccupied_point(local_point_cloud, n_samples):
    """
    sample unoccupied points along rays in local point cloud
    sensor located at origin
    local_point_cloud: <BxNxk>
    n_samples: number of samples on each ray
    """
    bs, L, k = local_point_cloud.shape
    unoccupied = torch.zeros(bs, L * n_samples, k,
                             device=local_point_cloud.device)
    for idx in range(1, n_samples + 1):
        fac = torch.rand(1).item()
        unoccupied[:, (idx - 1) * L:idx * L, :] = local_point_cloud * fac
    return unoccupied

class DeepMapping2D(nn.Module):
    def __init__(self, loss_fn, n_obs=256, n_samples=19, dim=[2, 64, 512, 512, 256, 128, 1]):
        super(DeepMapping2D, self).__init__()
        self.n_obs = n_obs
        self.n_samples = n_samples
        self.loss_fn = loss_fn
        self.loc_net = LocNetReg(n_points=n_obs, out_dims=3)
        self.occup_net = MLP(dim)

    def forward(self, obs_local,valid_points):
        # obs_local: <BxLx2>
        self.obs_local = deepcopy(obs_local)
        self.valid_points = valid_points

        self.pose_est = self.loc_net(self.obs_local)

        self.obs_global_est = transform_to_global_2D(
            self.pose_est, self.obs_local)

        if self.training:
            self.unoccupied_local = sample_unoccupied_point(
                self.obs_local, self.n_samples)
            self.unoccupied_global = transform_to_global_2D(
                self.pose_est, self.unoccupied_local)

            inputs, self.gt = get_M_net_inputs_labels(
                self.obs_global_est, self.unoccupied_global)
            self.occp_prob = self.occup_net(inputs)
            loss = self.compute_loss()
            return loss

    def compute_loss(self):
        valid_unoccupied_points = self.valid_points.repeat(1, self.n_samples)
        bce_weight = torch.cat(
            (self.valid_points, valid_unoccupied_points), 1).float()
        # <Bx(n+1)Lx1> same as occp_prob and gt
        bce_weight = bce_weight.unsqueeze(-1)

        if self.loss_fn.__name__ == 'bce_ch':
            loss = self.loss_fn(self.occp_prob, self.gt, self.obs_global_est,
                                self.valid_points, bce_weight, seq=4, gamma=0.1)  # BCE_CH
        elif self.loss_fn.__name__ == 'bce':
            loss = self.loss_fn(self.occp_prob, self.gt, bce_weight)  # BCE
        return loss
