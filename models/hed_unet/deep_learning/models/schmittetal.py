import numpy as np
import rasterio as rio
import yaml
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from math import pi, sqrt
from scipy.ndimage import gaussian_filter

from ..stats import lee_filter, anisodiff


class KMedians():
    def __init__(self, K, max_iter=100, tol=1e-4):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def _e_step(self, X):
        diff = X.unsqueeze(1) - self.cluster_centers.unsqueeze(0)
        self.labels = torch.sum(torch.abs(diff), axis=2).argmin(axis=1)

    def _average(self, X):
        return torch.median(X, axis=0)[0]

    def _m_step(self, X):
        X_center = None
        for center_id in range(self.K):
            center_mask = self.labels == center_id
            if not torch.any(center_mask):
                # The centroid of empty clusters is set to the center of
                # everything
                if X_center is None:
                    X_center = self._average(X)
                self.cluster_centers[center_id] = X_center
            else:
                self.cluster_centers[center_id] = \
                    self._average(X[center_mask])

    def fit(self, X):
        n_samples = X.shape[0]
        vdata = torch.mean(torch.var(X, axis=0))

        self.cluster_centers = X[torch.randperm(n_samples)[:self.K]]

        for i in range(self.max_iter):
            centers_old = self.cluster_centers.clone()

            self._e_step(X)
            self._m_step(X)

            if torch.sum((centers_old - self.cluster_centers) ** 2) < self.tol * vdata:
                break

        return self


class Schmittetal():
    tasks = ['seg']

    def __call__(self, data, *args, **kwargs):
        dev = data.device
        shp = data.shape[2:]
        data = F.avg_pool2d(data, 2)
        data = torch.log(data + 1e-8)
        data = data[0].numpy()
        ret = np.zeros_like(data[0])
        value_mask = (data != 0).all(axis=0)
        masks = []
        dilutions = [0, 1, 4, 16, 64] #, 256, 1024, 4096]
        for dilution in dilutions:
            if dilution > 0:
                img = np.stack([gaussian_filter(channel, sigma=dilution)
                    for channel in data], axis=-1)
            else:
                img = data.transpose(1, 2, 0)
            img = img[value_mask]

            kmed = KMedians(K=2)
            kmed.fit(torch.from_numpy(img).to(dev))

            cluster_centers = kmed.cluster_centers.cpu().numpy()
            labels = kmed.labels.cpu().numpy()

            land_class = cluster_centers[:, 0].argmax()
            masks.append(labels == land_class)
        mask = np.stack(masks, axis=-1).mean(axis=-1)
        ret[value_mask] = 2 * mask - 1
        ret = torch.from_numpy(ret.reshape(1, 1, *ret.shape))
        return F.interpolate(ret, size=shp, mode='bilinear')
