from .chamfer_dist import chamfer_loss
from .bce_loss import bce


def bce_ch(pred, targets, obs_global, valid_obs=None, bce_weight=None, seq=2, gamma=0.1):
    """
    pred: <Bx(n+1)Lx1>, occupancy probabiliry from M-Net
    targets: <Bx(n+1)Lx1>, occupancy label
    obs_global: <BxLxk> k = 2,3, global point cloud
    valid_obs: <BxL>, indices of valid point (0/1), 
               invalid points are not used for computing chamfer loss
    bce_weight: <Bx(n+1)Lx1>, weight for each point in computing bce loss
    """
    bce_loss = bce(pred, targets, bce_weight)
    ch_loss = chamfer_loss(obs_global, valid_obs, seq)
    return gamma * bce_loss + (1 - gamma) * ch_loss
