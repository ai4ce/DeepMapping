import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def get_and_init_FC_layer(din, dout):
    li = nn.Linear(din, dout)
    nn.init.xavier_uniform_(
       li.weight.data, gain=nn.init.calculate_gain('relu'))
    li.bias.data.fill_(0.)
    return li


def get_MLP_layers(dims, doLastRelu):
    layers = []
    for i in range(1, len(dims)):
        layers.append(get_and_init_FC_layer(dims[i - 1], dims[i]))
        if i == len(dims) - 1 and not doLastRelu:
            continue
        layers.append(nn.ReLU())
    return layers


class PointwiseMLP(nn.Sequential):
    def __init__(self, dims, doLastRelu=False):
        layers = get_MLP_layers(dims, doLastRelu)
        super(PointwiseMLP, self).__init__(*layers)


class MLP(nn.Module):
    def __init__(self, dims):
        super(MLP, self).__init__()
        self.mlp = PointwiseMLP(dims, doLastRelu=False)

    def forward(self, x):
        return self.mlp.forward(x)


class ObsFeat2D(nn.Module):
    """Feature extractor for 1D organized point clouds"""

    def __init__(self, n_points, n_out=1024):
        super(ObsFeat2D, self).__init__()
        self.n_out = n_out
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.conv1 = nn.Conv1d(2, 64, kernel_size=k, padding=p, dilation=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=k, padding=p, dilation=3)
        self.conv3 = nn.Conv1d(
            128, self.n_out, kernel_size=k, padding=p, dilation=3)
        self.mp = nn.MaxPool1d(n_points)

    def forward(self, x):
        assert(x.shape[1] == 2), "the input size must be <Bx2xL> "

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.mp(x)
        x = x.view(-1, self.n_out)  # <Bx1024>
        return x


class ObsFeatAVD(nn.Module):
    """Feature extractor for 2D organized point clouds"""
    def __init__(self, n_out=1024):
        super(ObsFeatAVD, self).__init__()
        self.n_out = n_out
        k = 3
        p = int(np.floor(k / 2)) + 2
        self.conv1 = nn.Conv2d(3,64,kernel_size=k,padding=p,dilation=3)
        self.conv2 = nn.Conv2d(64,128,kernel_size=k,padding=p,dilation=3)
        self.conv3 = nn.Conv2d(128,256,kernel_size=k,padding=p,dilation=3)
        self.conv4 = nn.Conv2d(256,self.n_out,kernel_size=k,padding=p,dilation=3)
        self.amp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        assert(x.shape[1]==3),"the input size must be <Bx3xHxW> "
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        x = self.amp(x) 
        x = x.view(-1,self.n_out) #<Bxn_out>
        return x


class LocNetReg2D(nn.Module):
    def __init__(self, n_points, out_dims):
        super(LocNetReg2D, self).__init__()
        self.obs_feat_extractor = ObsFeat2D(n_points)
        n_in = self.obs_feat_extractor.n_out
        self.fc = MLP([n_in, 512, 256, out_dims])

    def forward(self, obs):
        obs = obs.transpose(1, 2)
        obs_feat = self.obs_feat_extractor(obs)
        obs = obs.transpose(1, 2)

        x = self.fc(obs_feat)
        return x


class LocNetRegAVD(nn.Module):
    def __init__(self, out_dims):
        super(LocNetRegAVD, self).__init__()
        self.obs_feat_extractor = ObsFeatAVD()
        n_in = self.obs_feat_extractor.n_out
        self.fc = MLP([n_in, 512, 256, out_dims])

    def forward(self, obs):
        # obs: <BxHxWx3>
        bs = obs.shape[0]
        obs = obs.permute(0,3,1,2) # <Bx3xHxW>
        obs_feat = self.obs_feat_extractor(obs)
        obs = obs.permute(0,2,3,1)

        x = self.fc(obs_feat)
        return x
