import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..stats import lee_filter, anisodiff

class Lee():
    tasks = ['edge']
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

        self.gauss = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        self.gauss.weight.requires_grad = False
        for y in range(5):
            for x in range(5):
                dy = y - 2
                dx = x - 2
                self.gauss.weight[0, 0, y, x] = np.exp(-(dx*dx+dy*dy)/7)
        self.gauss.weight.set_(self.gauss.weight / self.gauss.weight.sum())

        self.roberts = nn.Conv2d(1, 2, 2, bias=False)
        self.roberts.weight.requires_grad = False
        self.roberts.weight.set_(torch.Tensor([[
            [-1,  0],
            [ 0,  1]],
           [[ 0, -1],
            [ 1,  0],
        ]]).reshape(2, 1, 2, 2))


    def __call__(self, data, *args, **kwargs):
        dev = data.device
        self.sobel = self.sobel.to(dev)
        self.roberts = self.roberts.to(dev)

        shp = data.shape[2:]
        data = data[:, [0]]
        inval_mask = (data == 0)
        data = torch.log(data + 1e-8)
        data = data.numpy()
        data = data[0, 0]
        data = lee_filter(data, 5)
        data = lee_filter(data, 5)
        data = lee_filter(data, 5)
        data = lee_filter(data, 5)
        data = torch.from_numpy(data).to(dev)
        data = data.unsqueeze(0).unsqueeze(0)

        dilated = data
        dilated = F.avg_pool2d(dilated, 5, padding=2, stride=1)
        edge = torch.linalg.norm(self.sobel(dilated), dim=1, keepdim=True)

        edge = ((edge - edge.min()) * 2) - 2
        edge[inval_mask] = -100

        return edge
