import numpy as np
import rasterio as rio
import yaml
import matplotlib.pyplot as plt
import torch
from math import pi, sqrt

from ..stats import lee_filter, anisodiff

def pdf_twoparts(x, a, b, mu1, mu2, lv1, lv2):
    A = torch.exp(a)
    B = torch.exp(b)
    sigma1 = torch.exp(lv1)
    sigma2 = torch.exp(lv2)

    left  = A / sigma1 * torch.exp(-torch.square(x - mu1) / (2 * torch.square(sigma1)))
    right = B / sigma2 * torch.exp(-torch.square(x - mu2) / (2 * torch.square(sigma2)))

    return left, right

def pdf(x, a, b, mu1, mu2, lv1, lv2):
    left, right = pdf_twoparts(x, a, b, mu1, mu2, lv1, lv2)
    return left + right

class LiuJezek():
    tasks = ['seg']

    def __call__(self, data, *args, **kwargs):
        if type(data) is torch.Tensor:
            data = data.numpy()
        data = data[0, 0]
        val_mask = (data != 0) & (data < 3)
        data = lee_filter(data, 5)
        data = anisodiff(data, niter=5, kappa=8, gamma=0.25)
        vals = data[val_mask]

        x = np.linspace(0, 3, 1000)
        counts = np.bincount((vals * 1000 / 3).astype(np.int64), minlength=1000)[:1000]
        hist = torch.from_numpy(counts / counts.sum()).to(torch.float32)
        x = torch.from_numpy(x).to(torch.float32)

        with torch.enable_grad():
            params = torch.Tensor([-6, -6, 1, 2, -2, -2])
            params.requires_grad = True

            opt = torch.optim.SGD([params], lr=1e-1, momentum=0.9)
            for i in range(2000):
                state = pdf(x, *params)
                loss = torch.sum(torch.square(state - hist))
                opt.zero_grad()
                loss.backward()
                opt.step()

            A, B, mu1, mu2, lv1, lv2 = params
            assert mu1 < mu2
            left, right = pdf_twoparts(x, *params.detach())
            T = x[np.argmax(left < right)].numpy()
            res = torch.from_numpy(data > T).unsqueeze(0).unsqueeze(0).to(torch.float32)
            return 2 * res - 1
