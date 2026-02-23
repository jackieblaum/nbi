from functools import partial

import numpy as np
from tqdm import tqdm
import torch


def parallel_simulate(args):
    """
    thetas, paths, simulator are packaged into args because this function will be called using multiprocessing

    :param args: thetas, paths, simulator
        thetas: shape = (num_simulations, dim_parameters)
        paths: shape = (num_simulations,)
        simulator: callable.
    :return: mask of good simulations
    """

    thetas, paths, simulator = args
    use_tqdm = "/0.npy" in paths[0]
    mask = []
    if use_tqdm:
        print("Generating simulations")
    for i, params in tqdm(enumerate(thetas), disable=not use_tqdm):
        simulation = simulator(params)
        np.save(paths[i], simulation)
        mask.append(not np.isnan(simulation).any() and not np.isinf(simulation).any())

    return mask


def add_noise(x_err, x, y=None):
    """
    x: light curve of shape (length,)
    y: parameter of shape (dim,)
    """
    rand = np.random.normal(0, 1, size=x.shape[0])
    x_noise = x + rand * x_err
    return x_noise, y


def iid_gaussian(x_err):
    return partial(add_noise, x_err)


def log_like(x_err, x, x_path, y):
    # x is observed data, x_path is path to saved model prediction
    model = np.load(x_path)
    chi2 = (((x - model) / x_err) ** 2).sum()
    return -chi2 / 2


def log_like_iidg(x_err):
    return partial(log_like, x_err)

def collate_list_channels(batch):
    """
    Batch is a list of tuples: [(x_list, y), ...]
    where x_list is list of tensors: [x0, x1, ..., xK]
    Returns:
    x_batched: list of tensors, each stacked to [B, ...]
    y_batched: tensor [B, D]
    """
    xs, ys = zip(*batch)  # xs is tuple of lists
    k = len(xs[0])
    x_out = [torch.stack([x_i[j] for x_i in xs], dim=0) for j in range(k)]
    y_out = torch.stack(list(ys), dim=0)
    return x_out, y_out


def masked_mean_std(xs, mask_dim, masked_dims):
    """
    xs: np.ndarray [N, C, L]
    mask_dim: int
    masked_dims: list[int]
    Returns mu, sd shaped [1, C, 1]
    """
    C = xs.shape[1]
    mu = np.zeros((1, C, 1), dtype=np.float32)
    sd = np.ones((1, C, 1), dtype=np.float32)

    mask = xs[:, mask_dim, :]  # [N, L]
    # force mask to {0,1} and finite
    mask = np.where(np.isfinite(mask), mask, 0.0)
    mask = (mask > 0.5).astype(np.float32)

    for d in masked_dims:
        v = xs[:, d, :]
        # treat non-finite values as missing
        finite = np.isfinite(v)
        w = mask * finite.astype(np.float32)
        wsum = w.sum()

        if wsum <= 0:
            # fallback: unmasked finite-only stats
            vv = v[finite]
            if vv.size == 0:
                mu[0, d, 0] = 0.0
                sd[0, d, 0] = 1.0
                continue
            m = float(vv.mean())
            s = float(vv.std())
            if (not np.isfinite(s)) or s == 0:
                s = 1.0
            mu[0, d, 0] = m
            sd[0, d, 0] = s
            continue

        m = (v * w).sum() / wsum
        var = ((v - m) ** 2 * w).sum() / wsum
        s = np.sqrt(var)

        if not np.isfinite(m):
            m = 0.0
        if not np.isfinite(s) or s == 0:
            s = 1.0

        mu[0, d, 0] = float(m)
        sd[0, d, 0] = float(s)

    return mu, sd