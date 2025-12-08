import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..stats import lee_filter, anisodiff
from skimage import segmentation, feature, filters


class ActiveContour():
    tasks = ['edge']
    patchsize = 768

    def __init__(self):
        self.sobel = nn.Conv2d(1, 2, 3, padding=1, padding_mode='replicate', bias=False)
        self.sobel.weight.requires_grad = False
        self.sobel.weight.set_(torch.Tensor([[
            [-1,  0,  1],
            [-2,  0,  2],
            [-1,  0,  1]],
           [[-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ]]).reshape(2, 1, 3, 3))

    def __call__(self, data, *args, **kwargs):
        dev = data.device
        self.sobel = self.sobel.to(dev)

        shp = data.shape[2:]
        data = torch.log(data + 1e-8)
        data = F.avg_pool2d(data, 4)
        data = data[0, 0].cpu().numpy()
        seg = segmentation.chan_vese(data, mu=0.2, max_iter=200)
        seg = seg.reshape(1, 1, *seg.shape)
        seg = F.interpolate(torch.from_numpy(seg).to(dev, torch.float), size=shp, mode='bilinear')
        edge = torch.linalg.norm(self.sobel(seg), dim=1, keepdims=True)
        edge = ((edge - edge.min()) * 2000) - 1000
        return edge
