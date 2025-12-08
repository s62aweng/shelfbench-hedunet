import torch
import torch.nn as nn
import torch.nn.functional as F
from .marching_squares import MarchingSquares
from ..layers import DownBlock, Convx2
from math import sqrt


class SnakeBlock(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(input_channels, 2, 1, bias=False)
        )
        with torch.no_grad():
            self.fc[2].weight[:] = 0

    def forward(self, features, batch_edges, edges_per_img):
        if batch_edges.shape[0] == 0:
            return batch_edges
        return_edges = []
        single_edges = batch_edges.split(edges_per_img)
        for b, edges in enumerate(single_edges):
            # TODO handle non-square inputs
            # Align points to [-1, -1, 1, 1]
            if edges.shape[0] == 0:
                continue
            points = edges.reshape(-1, 2) * (2 / features.shape[2]) - 1
            values = features[b].unsqueeze(0)
            points = points.unsqueeze(0).unsqueeze(1)
            sampled = F.grid_sample(values, points, align_corners=False)
            sampled = sampled.squeeze(2)
            diffs = self.fc(sampled)
            diffs = diffs.squeeze(0).transpose(0, 1).reshape(-1, 4)
            return_edges.append(edges + diffs)
        return torch.cat(return_edges, dim=0)


class SnakeNet(nn.Module):
    def __init__(self, input_channels, base_channels=16):
        super().__init__()
        bc = base_channels

        # Backbone
        self.backbone = nn.Sequential(
            Convx2(input_channels, 1 * bc, bn=True),
            DownBlock( 1 * bc,  2 * bc),
            DownBlock( 2 * bc,  4 * bc),
            DownBlock( 4 * bc,  8 * bc),
            DownBlock( 8 * bc, 16 * bc),
            DownBlock(16 * bc, 32 * bc),
        )

        self.make_coarse = nn.Conv2d(32 * bc, 1, 1)
        self.marching_squares = MarchingSquares()
        self.snake_block = SnakeBlock(32 * bc)

        self.pretraining_phase = True

    def forward(self, x):
        features = self.backbone(x)
        coarse = self.make_coarse(features)

        if self.pretraining_phase:
            return None, coarse

        # Do Marching squares
        edgestack = []
        edges, edges_per_img = self.marching_squares(coarse)
        edgestack.append(edges)
        # `edges` and `features` now live in the same downsampled feature space
        edges = self.snake_block(features, edges, edges_per_img)
        edgestack.append(edges)
        edges = self.snake_block(features, edges, edges_per_img)
        edgestack.append(edges)

        return (edgestack, edges_per_img), coarse


class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        SQ = sqrt(2)
        self.register_buffer('make_normal', torch.Tensor([
            [ 0,  1,  0,  -1],
            [-1,  0,  1,   0],
            [-SQ, -SQ,  SQ,  SQ],
            [ SQ, -SQ, -SQ,  SQ],
            [ SQ, -SQ, -SQ,  SQ],
            [ SQ,  SQ, -SQ, -SQ],
        ]).T)

    def forward(self, edges, target):
        normals = torch.matmul(edges, self.make_normal)
        normals = normals.view(-1, 3, 2) # split up the 3 normals

        sizes = torch.pow(normals, 2).sum(dim=2, keepdims=True)

        normals = normals / sizes.sqrt()
        normals = normals.view(-1, 6) # rejoin the 3 normals

        dists_left  = (edges[:, [0, 1, 0, 1, 2, 3]] * normals).view(-1, 3, 2).sum(dim=2)
        dists_right = (edges[:, [0, 1, 2, 3, 0, 1]] * normals).view(-1, 3, 2).sum(dim=2)

        ly = torch.arange(target.shape[1], device=target.device).float()
        lx = torch.arange(target.shape[2], device=target.device).float()
        grid = torch.stack(torch.meshgrid(ly, lx), dim=-1)

        # Expand everything along to Batch x edge x HW x xy
        bc_grid = grid.view(-1, 2).unsqueeze(0).unsqueeze(0)
        bc_normals = normals.view(-1, 3, 2, 1)
        bc_dists_left = dists_left.unsqueeze(2).unsqueeze(3)
        bc_dists_right = dists_right.unsqueeze(2).unsqueeze(3)

        dotprods = torch.matmul(bc_grid, bc_normals).squeeze(2)

        sdf = dotprods - bc_dists_left
        sdf = sdf.view(-1, 3, *target.shape)
        sdf = torch.min(sdf, dim=1)[0]
        leftness = torch.sigmoid(sdf) / sizes[:, [0]]

        sdf = -(dotprods - bc_dists_right)
        sdf = sdf.view(-1, 3, *target.shape)
        sdf = torch.min(sdf, dim=1)[0]
        rightness = torch.sigmoid(sdf) / sizes[:, [0]]

        # target = target.unsqueeze(0)
        # loss = -insideness * target - outsideness * (1 - target)
        loss = (-leftness * target - rightness * (1 - target))

        return loss.mean()


class SnakeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.edgeloss = EdgeLoss()

    def forward(self, pred, target):
        N, C, H, W = target.shape
        edge_info, coarse = pred
        coarse_target = target.view(N, C, H // 32, 32, W // 32, 32)
        coarse_target = coarse_target.mean(axis=-1).mean(axis=-2)
        coarse_loss = F.binary_cross_entropy_with_logits(coarse, coarse_target)

        self.cached_coarse_loss = coarse_loss

        if edge_info is None or edge_info[0][0].shape[0] == 0:
            return coarse_loss
        all_edges, edges_per_img = edge_info

        edgelosses = []
        for edgeset in all_edges:
            unbatched = edgeset.split(edges_per_img)
            for edges, target_slice in zip(unbatched, coarse_target):
                if edges.shape[0] == 0:
                    continue
                edgelosses.append(self.edgeloss(edges, target_slice))

        full_edgeloss = torch.mean(torch.stack(edgelosses))
        self.cached_edgeloss = full_edgeloss

        return coarse_loss + 0.01 * full_edgeloss
