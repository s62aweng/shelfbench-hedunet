import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarchingSquares(nn.Module):
    def __init__(self):
        super().__init__()
        self.corner_points = np.array([[.5, -.5], [.5, .5], [-.5, .5], [-.5, -.5]], np.float32)[np.newaxis]
        self.register_buffer('coefficients', torch.Tensor([
            [ -1,  0,  1,  0,  0,  0,  0,  0],
            [  0, -1,  0,  1,  0,  0,  0,  0],
            [  0,  0,  0,  0, -1,  0,  1,  0],
            [  0,  0,  0,  0,  0, -1,  0,  1],
        ]).t().contiguous())

        SW, SE, NE, NW = range(4)
        self.first_segment = np.array([
            [-1, -1, -1, -1],
            [NW, SW, SE, SW],
            [SW, SE, NE, SE],
            [NW, SW, NE, SE],
            [SE, NE, NW, NE],
            [NW, SW, SE, SW],
            [SW, SE, NW, NE],
            [NW, SW, NW, NE],
            [NE, NW, SW, NW],
            [NE, NW, SE, SW],
            [SW, SE, NE, SE],
            [NE, NW, NE, SE],
            [SE, NE, SW, NW],
            [SE, NE, SE, SW],
            [SW, SE, SW, NW],
        ])
        self.second_segment = np.array([
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [SE, NE, NW, NE],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [-1, -1, -1, -1],
            [NE, NW, SW, NW],
        ])

    def forward(self, coarse):
        # Replicate Padding ensures we don't create edges parallel to the border
        coarse = F.pad(coarse, [1, 1, 1, 1], mode='replicate').squeeze(1)
        sw = coarse[:, 1:, :-1]
        se = coarse[:, 1:, 1:]
        nw = coarse[:, :-1, :-1]
        ne = coarse[:, :-1, 1:]
        values = torch.stack([sw, se, ne, nw], dim=-1)

        # Indexing calculations done on cpu (easier, faster, and they don't need grad)
        cells = (values > 0).to('cpu', torch.bool).numpy()
        cellidx = (np.packbits(cells, axis=3, bitorder='little')).squeeze(3)

        E1B, E1Y, E1X = np.nonzero((cellidx > 0) & (cellidx < 15))
        E2B, E2Y, E2X = np.nonzero((cellidx == 0b1010) | (cellidx == 0b0101))
        cell_batch = np.hstack([E1B, E2B])
        reordered = np.argsort(cell_batch, kind='stable')
        cell_batch = cell_batch[reordered]
        edges_per_img = tuple(np.bincount(cell_batch, minlength=coarse.shape[0]))

        required_vals = np.concatenate([
            self.first_segment[cellidx[E1B, E1Y, E1X]],
            self.second_segment[cellidx[E2B, E2Y, E2X]]
        ], axis=0)[reordered]

        required_vals = torch.from_numpy(required_vals).to(coarse.device, non_blocking=True)

        cell_y = np.hstack([E1Y, E2Y])[reordered]
        cell_x = np.hstack([E1X, E2X])[reordered]
        cell_yx = np.hstack([cell_y[:, np.newaxis], cell_x[:, np.newaxis]])

        cell_corners = torch.from_numpy(cell_yx[:, np.newaxis] + self.corner_points)
        cell_corners = cell_corners.to(coarse.device, torch.float32, non_blocking=True)

        vals = values[cell_batch, cell_y, cell_x]

        double_idx = torch.stack([required_vals, required_vals], dim=2)
        final_vals = torch.gather(vals, dim=1, index=required_vals).unsqueeze(2)
        final_corners = torch.gather(cell_corners, dim=1, index=double_idx)

        # assert (final_vals[:, :, ::2] <= 0).all(), \
        #         f"Only {(100 * final_vals[:, :, ::2] <= 0).float().mean()}% of lower vals <= 0"
        # assert (final_vals[:, :, 1::2] >= 0).all(), \
        #         f"Only {(100 * final_vals[:, :, 1::2] >= 0).float().mean()}% of upper vals >= 0"

        coords = (final_vals * final_corners).view(-1, 8)
        vals = torch.stack([final_vals, final_vals], dim=2).view(-1, 8)

        numerator = torch.matmul(coords, self.coefficients)
        denominator = torch.matmul(vals, self.coefficients)
        # FIXME Should be all positive...

        # return numerator / denominator, edges_per_img
        return torch.matmul(final_corners.reshape(-1, 8), torch.abs(self.coefficients)) / 2, edges_per_img

CELLMASK = torch.Tensor([1,2,4,8]).to(torch.short)

SW, SE, NE, NW = range(4)
SEGMENTS = [[] for _ in range(16)]
SEGMENTS[0b0000] = []
SEGMENTS[0b0001] = [[[NW, SW], [SE, SW]]]
SEGMENTS[0b0010] = [[[SW, SE], [NE, SE]]]
SEGMENTS[0b0011] = [[[NW, SW], [NE, SE]]]
SEGMENTS[0b0100] = [[[SE, NE], [NW, NE]]]
SEGMENTS[0b0101] = [[[NW, SW], [SE, SW]], [[SE, NE], [NW, NE]]]
SEGMENTS[0b0110] = [[[SW, SE], [NW, NE]]]
SEGMENTS[0b0111] = [[[NW, SW], [NW, NE]]]
SEGMENTS[0b1000] = [[[NE, NW], [SW, NW]]]
SEGMENTS[0b1001] = [[[NE, NW], [SE, SW]]]
SEGMENTS[0b1010] = [[[SW, SE], [NE, SE]], [[NE, NW], [SW, NW]]]
SEGMENTS[0b1011] = [[[NE, NW], [NE, SE]]]
SEGMENTS[0b1100] = [[[SE, NE], [SW, NW]]]
SEGMENTS[0b1101] = [[[SE, NE], [SE, SW]]]
SEGMENTS[0b1110] = [[[SW, SE], [SW, NW]]]
SEGMENTS[0b1111] = []

def legacy_ms(coarse):
    coarse = F.pad(coarse, [1, 1, 1, 1], mode='replicate')
    coarse_class = coarse > 0
    nw = coarse[:, :, :-1, :-1]
    ne = coarse[:, :, :-1, 1:]
    sw = coarse[:, :, 1:, :-1]
    se = coarse[:, :, 1:, 1:]
    values = torch.stack([sw, se, ne, nw], dim=-1)

    cells = (values > 0).to(torch.short)
    cellidx = torch.tensordot(cells, CELLMASK, dims=([-1], [0]))

    edges_per_img = []
    edges = []
    for b in range(cells.shape[0]):
        edges_written = 0
        for y in range(cells.shape[2]):
            for x in range(cells.shape[3]):
                idx = cellidx[b, 0, y, x]
                if len(SEGMENTS[idx]) == 0:
                    continue
                segments = SEGMENTS[idx]
                persistent_idx = idx
                vals = values[b, 0, y, x]
                points = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]]) + torch.tensor([[y, x]]) - 0.5
                for (l1, u1), (l2, u2) in segments:
                    assert vals[u1] >= 0
                    assert vals[u2] >= 0
                    assert vals[l1] <= 0
                    assert vals[l2] <= 0
                    start = (vals[u1] * points[u1] - vals[l1] * points[l1]) / (vals[u1] - vals[l1])
                    stop  = (vals[u2] * points[u2] - vals[l2] * points[l2]) / (vals[u2] - vals[l2])
                    edge = torch.cat([start, stop], dim=0)
                    edges.append(edge)
                    edges_written += 1
        edges_per_img.append(edges_written)
        edges_written = 0
    return torch.stack(edges, dim=0), tuple(edges_per_img)
