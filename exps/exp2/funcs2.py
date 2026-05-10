# funcs2.py
# GPU-enabled OT experiment utilities

import torch
import ot
import time
import pickle
import os
from tqdm import tqdm


# ===========================================================
# Device setup
# ===========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===========================================================
# Distribution generator
# ===========================================================
def generate_distribution(name, n_samples, dim, center=True, device=device):
    """
    Generate samples from a given distribution.
    """

    if name == "Gaussian":
        X = torch.randn((n_samples, dim), device=device)

    elif name == "Uniform":
        X = torch.rand((n_samples, dim), device=device) * 2 - 1

    elif name == "Poisson":
        base = torch.poisson(torch.ones((n_samples, 1), device=device))
        X = base.repeat(1, dim)

    elif name == "Geometric":
        p = 0.5
        u = torch.rand((n_samples, 1), device=device)
        base = torch.ceil(torch.log(1 - u) / torch.log(torch.tensor(1 - p, device=device)))
        X = base.repeat(1, dim)

    elif name == "Gamma":
        X = torch.distributions.Gamma(1., 1.).sample((n_samples, dim)).to(device)

    else:
        raise ValueError(f"Unknown distribution: {name}")

    if center:
        X -= X.mean(dim=0, keepdim=True)

    return X



# ===========================================================
# Cost normalization
# ===========================================================
def normalize_cost(C):
    return C / C.max()


# ===========================================================
# OT Solvers
# ===========================================================

def sinkhorn_w2(X, Y, a, b, reg, epsilon, max_iter):
    """
    Standard Sinkhorn
    """
    C = torch.cdist(X, Y, p=2) ** 2

    return ot.sinkhorn2(
        a, b, C,
        reg=reg,
        stopThr=epsilon,
        numItermax=max_iter,
        method="sinkhorn_log"
    )


def sinkhorn_w2_normalized(X, Y, a, b, reg, epsilon, max_iter):
    """
    Sinkhorn with cost normalization
    C <- C / max(C)
    """
    C = torch.cdist(X, Y, p=2) ** 2
    Cn = normalize_cost(C)

    return ot.sinkhorn2(
        a, b, Cn,
        reg=reg,
        stopThr=epsilon,
        numItermax=max_iter,
        method="sinkhorn_log"
    ) * C.max()


def rw2_sinkhorn_w2(X, Y, a, b, reg, epsilon, max_iter):
    """
    RW2 Sinkhorn (centering)
    """

    mean_X = torch.matmul(X.T, a)
    mean_Y = torch.matmul(Y.T, b)

    C = torch.cdist(X, Y, p=2) ** 2
    C_shift = torch.cdist(X - mean_X, Y - mean_Y, p=2) ** 2

    maxC = C.max().item()
    maxC_shift = C_shift.max().item()

    if maxC_shift <= maxC:
        return ot.sinkhorn2(
            a, b, C_shift,
            reg=reg,
            stopThr=epsilon,
            numItermax=max_iter,
            method="sinkhorn_log"
        ) + torch.norm(mean_X - mean_Y) ** 2

    else:
        return ot.sinkhorn2(
            a, b, C,
            reg=reg,
            stopThr=epsilon,
            numItermax=max_iter,
            method="sinkhorn_log"
        )


def rw2_sinkhorn_w2_normalized(X, Y, a, b, reg, epsilon, max_iter):
    """
    RW2 + cost normalization
    """

    mean_X = torch.matmul(X.T, a)
    mean_Y = torch.matmul(Y.T, b)

    C_shift = torch.cdist(X - mean_X, Y - mean_Y, p=2) ** 2
    Cn = normalize_cost(C_shift)

    return ot.sinkhorn2(
        a, b, Cn,
        reg=reg,
        stopThr=epsilon,
        numItermax=max_iter,
        method="sinkhorn_log"
    ) * C_shift.max() + torch.norm(mean_X - mean_Y) ** 2


def exact_emd(X, Y, a, b):
    """
    Exact EMD (ground truth)
    """
    C = torch.cdist(X, Y, p=2) ** 2
    return ot.emd2(a, b, C)


# ===========================================================
# Utility functions
# ===========================================================

def compute_mean_std_by_col(arr):

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()

    mean = arr.mean(axis=0)
    std = arr.std(axis=0)

    return mean, std


def make_filename(source_dist, target_dist, dim, size, reg,
                  stopThr, max_iter, repeated_times, t_list):

    t_str = "_".join([str(t) for t in t_list])

    return (
        f"dim{dim}_size{size}_reg{reg}_eps{stopThr}"
        f"_iter{max_iter}_rep{repeated_times}_t{t_str}.pkl"
    )


# ===========================================================
# Main experiment runner
# ===========================================================

def run_experiment(
        source_dist, target_dist,
        dims, sizes,
        t_list,
        reg=1e-3,
        stopThr=1e-3,
        max_iter=1000,
        repeated_times=2,
        cache_root="results"
):

    pair_dir = os.path.join(cache_root, f"{source_dist}_vs_{target_dist}")
    os.makedirs(pair_dir, exist_ok=True)

    results = {}

    total_iters = len(dims) * len(sizes) * repeated_times * len(t_list)

    with tqdm(total=total_iters, desc="Total experiment progress") as global_pbar:

        for dim in dims:
            for size in sizes:

                filename = os.path.join(
                    pair_dir,
                    make_filename(
                        source_dist, target_dist,
                        dim, size, reg, stopThr,
                        max_iter, repeated_times, t_list
                    )
                )

                if os.path.exists(filename):

                    with open(filename, "rb") as f:
                        results[(source_dist, target_dist, dim, size)] = pickle.load(f)

                    print(f"Loaded cached results from {filename}")
                    global_pbar.update(repeated_times * len(t_list))
                    continue

                # Storage tensors
                time_sk = torch.zeros((repeated_times, len(t_list)))
                time_norm = torch.zeros((repeated_times, len(t_list)))
                time_rot = torch.zeros((repeated_times, len(t_list)))
                time_rot_norm = torch.zeros((repeated_times, len(t_list)))
                time_emd = torch.zeros((repeated_times, len(t_list)))

                w2_sk = torch.zeros((repeated_times, len(t_list)))
                w2_norm = torch.zeros((repeated_times, len(t_list)))
                w2_rot = torch.zeros((repeated_times, len(t_list)))
                w2_rot_norm = torch.zeros((repeated_times, len(t_list)))
                w2_emd = torch.zeros((repeated_times, len(t_list)))

                for r in range(repeated_times):

                    for ti, t in enumerate(t_list):

                        # Generate distributions
                        X = generate_distribution(source_dist, size, dim)
                        Y = generate_distribution(target_dist, size, dim)

                        Y[:, -1] += t

                        a = torch.ones(size, device=device) / size
                        b = torch.ones(size, device=device) / size

                        # Sinkhorn
                        tic = time.time()
                        w2_sk[r, ti] = sinkhorn_w2(X, Y, a, b, reg, stopThr, max_iter)
                        time_sk[r, ti] = time.time() - tic

                        # Sinkhorn normalized
                        tic = time.time()
                        w2_norm[r, ti] = sinkhorn_w2_normalized(X, Y, a, b, reg, stopThr, max_iter)
                        time_norm[r, ti] = time.time() - tic

                        # RW2
                        tic = time.time()
                        w2_rot[r, ti] = rw2_sinkhorn_w2(X, Y, a, b, reg, stopThr, max_iter)
                        time_rot[r, ti] = time.time() - tic

                        # RW2 + normalization
                        tic = time.time()
                        w2_rot_norm[r, ti] = rw2_sinkhorn_w2_normalized(
                            X, Y, a, b, reg, stopThr, max_iter
                        )
                        time_rot_norm[r, ti] = time.time() - tic

                        # Exact EMD
                        tic = time.time()
                        w2_emd[r, ti] = exact_emd(X, Y, a, b)
                        time_emd[r, ti] = time.time() - tic

                        global_pbar.update(1)

                res = {
                    "time": {
                        "sk": time_sk.cpu().numpy(),
                        "norm": time_norm.cpu().numpy(),
                        "rot": time_rot.cpu().numpy(),
                        "rot_norm": time_rot_norm.cpu().numpy(),
                        "emd": time_emd.cpu().numpy(),
                    },
                    "w2": {
                        "sk": w2_sk.cpu().numpy(),
                        "norm": w2_norm.cpu().numpy(),
                        "rot": w2_rot.cpu().numpy(),
                        "rot_norm": w2_rot_norm.cpu().numpy(),
                        "emd": w2_emd.cpu().numpy(),
                    },
                    "t_list": t_list,
                }

                results[(source_dist, target_dist, dim, size)] = res

                with open(filename, "wb") as f:
                    pickle.dump(res, f)

                print(f"Saved results to {filename}")

    return results


# ===========================================================
# Config sanitizer
# ===========================================================

def sanitize_config(exp):

    exp["dims"] = [int(d) for d in exp["dims"]]
    exp["sizes"] = [int(s) for s in exp["sizes"]]
    exp["t_list"] = [float(t) for t in exp["t_list"]]

    exp["reg"] = float(exp["reg"])
    exp["stopThr"] = float(exp["stopThr"])

    exp["max_iter"] = int(exp["max_iter"])
    exp["repeated_times"] = int(exp["repeated_times"])

    return exp