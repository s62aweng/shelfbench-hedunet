import numpy as np
from numba import jit, prange
from tqdm import trange, tqdm


@jit(nopython=True, parallel=True)
def make_sdf_numba(mask, truncation=500):
    """
    Calculates a truncated signed distance field on the given binary mask
    """
    _, H, W = mask.shape
    sdf = np.zeros((1, H, W), np.float32)

    offsets = []
    for y in range(-truncation, truncation):
        for x in range(-truncation, truncation):
            offsets.append((y, x))
    offsets.sort(key=lambda t: t[0]*t[0]+t[1]*t[1])

    for y in range(0, H):
        for x in prange(0, W):
            base = mask[0, y, x]

            lo_y = max(0, y - truncation)
            hi_y = min(H, y + 1 + truncation)  # exclusive
            lo_x = max(0, x - truncation)
            hi_x = min(W, x + 1 + truncation)  # exclusive

            best = truncation * truncation

            for dy, dx in offsets:
                y2 = y + dy
                x2 = y + dx
                if y2 < 0 or y2 >= H or x2 < 0 or x2 >= W:
                    continue
                if not base == mask[0, y2, x2]:
                    dy = y
                    dx = x - x2
                    best = min(best, dy * dy + dx * dx)
                    break

            best = np.sqrt(best)
            if not base:
                best = -best
            sdf[0, y, x] = best

    return sdf / truncation


def xsweep(img, out, direction):
    H, W = img.shape

    xiter = range(W)
    if direction == -1:
        xiter = range(W-1, -1, -1)

    dist = -np.ones(H)

    for x in tqdm(xiter):
        val = img[:, x]
        fg = val != 0
        mask = (dist < 0) & (~fg)
        dist[mask &  fg] = 0
        dist[mask & ~fg] += 1
        out[:, x] = np.minimum(out[:, x], dist ** 2)


def sdf(img):
    """Calcualates the SDF transform of a binary image

    Heavily based on https://github.com/JuliaGraphics/SignedDistanceFields.jl
    """

    assert img.ndim == 2
    H, W = img.shape

    upper_bound = img.shape[0]*img.shape[0]+img.shape[1]*img.shape[0]
    rowdf_sq = upper_bound * np.ones_like(img)

    print('XSweep Phase.')

    xsweep(img, rowdf_sq, 1)
    xsweep(img, rowdf_sq, -1)

    print('Done. Full Phase')

    df_sq = upper_bound * np.ones_like(img)

    for x in trange(W):
        coords = np.arange(H)
        dist = np.square(coords.reshape(-1, 1) - coords.reshape(1, -1))
        dist = dist + df_sq[:, [x]]
        df_sq[:, x] = dist.min(axis=0)

    print('Done.')

    return np.sqrt(df_sq)
