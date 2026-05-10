import torch
import ot
import time
import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.set_default_dtype(torch.float64)

# ===========================================================
# Matplotlib setup (paper-quality)
# ===========================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.5,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42,
})

# ===========================================================
# Color mapping (same pattern as before)
# ===========================================================
color_map = {
    2: {"lp": "tab:blue", "rw2": "tab:orange"},
    10: {"lp": "tab:green", "rw2": "tab:red"},
}

# ===========================================================
# Distribution generator
# ===========================================================
def generate_distribution(name, n_samples, dim, center=True):
    """Generate n_samples × dim samples from a named distribution."""
    if name == "Gaussian":
        X = torch.randn((n_samples, dim))
    elif name == "Uniform":
        X = torch.rand((n_samples, dim)) * 2 - 1
    elif name == "Poisson":
        base = torch.poisson(torch.ones((n_samples, 1)))
        X = base.repeat(1, dim)
    elif name == "Gamma":
        X = torch.distributions.Gamma(1.0, 1.0).sample((n_samples, dim))
    else:
        raise ValueError(f"Unknown distribution: {name}")

    if center:
        X -= X.mean(dim=0, keepdim=True)
    return X


# ===========================================================
# Helpers
# ===========================================================
def compute_mean_std(arr):
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    return arr.mean(axis=0), arr.std(axis=0)


def normalize_cost_matrix(C):
    """Normalize a cost matrix by its maximum entry."""
    cmax = C.max()
    if cmax <= 0:
        return C, 1.0
    return C / cmax, cmax


# ===========================================================
# LP solvers
# ===========================================================
def exact_lp(X, Y, a, b, itermax=1000):
    """Standard LP solver."""
    C = torch.cdist(X, Y, p=2) ** 2
    return ot.emd2(a, b, C, numItermax=itermax)


def exact_lp_norm(X, Y, a, b, itermax=1000):
    """Standard LP solver with normalization."""
    C = torch.cdist(X, Y, p=2) ** 2
    Cn, cmax = normalize_cost_matrix(C)
    val = ot.emd2(a, b, Cn, numItermax=itermax)
    return val * cmax


def rw2_lp(X, Y, a, b, itermax=1000, M=1.0):
    """
    RW2-LP solver: centered LP plus mean correction.
    Threshold parameter fixed at M=1.
    """
    mean_X = torch.matmul(X.T, a)
    mean_Y = torch.matmul(Y.T, b)

    Xc = X - mean_X
    Yc = Y - mean_Y

    C_shift = torch.cdist(Xc, Yc, p=2) ** 2
    C_orig = torch.cdist(X, Y, p=2) ** 2

    # threshold rule with M fixed at 1
    if C_shift.max() <= M * C_orig.max():
        return ot.emd2(a, b, C_shift, numItermax=itermax) + torch.norm(mean_X - mean_Y) ** 2
    else:
        return ot.emd2(a, b, C_orig, numItermax=itermax)


def rw2_lp_norm(X, Y, a, b, itermax=1000, M=1.0):
    """
    RW2-LP solver with normalization.
    Normalize the centered cost matrix by its maximum.
    """
    mean_X = torch.matmul(X.T, a)
    mean_Y = torch.matmul(Y.T, b)

    Xc = X - mean_X
    Yc = Y - mean_Y

    C_shift = torch.cdist(Xc, Yc, p=2) ** 2
    C_orig = torch.cdist(X, Y, p=2) ** 2

    if C_shift.max() <= M * C_orig.max():
        Cn, cmax = normalize_cost_matrix(C_shift)
        val = ot.emd2(a, b, Cn, numItermax=itermax) * cmax
        return val + torch.norm(mean_X - mean_Y) ** 2
    else:
        Cn, cmax = normalize_cost_matrix(C_orig)
        return ot.emd2(a, b, Cn, numItermax=itermax) * cmax


# ===========================================================
# Main Experiment Function
# ===========================================================
def run_experiment(
    source_dist="Gaussian",
    target_dist="Uniform",
    dims=[2, 10],
    sizes=[512],
    repeated_times=6,
    cache_root="results_gauss_uniform",
    save_dir="results_figs_gauss_uniform",
    gt_itermax=1_000_000_000,
    M=1.0,
):
    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    itermax_list = [2 ** i for i in range(6, 17)]
    results = {}
    total_iters = len(dims) * len(sizes) * repeated_times * len(itermax_list)

    with tqdm(total=total_iters, desc=f"Running EMD Experiments ({source_dist} → {target_dist})") as pbar:
        for dim in dims:
            for size in sizes:
                cache_file = os.path.join(
                    cache_root,
                    f"{source_dist}_to_{target_dist}_dim{dim}_size{size}_v2.pkl"
                )

                if os.path.exists(cache_file):
                    with open(cache_file, "rb") as f:
                        res = pickle.load(f)

                    # compatibility check
                    if "lp" in res["time"] and "lp_norm" in res["time"] and "rw2" in res["time"] and "rw2_norm" in res["time"]:
                        results[(dim, size)] = res
                        print(f"Loaded cached results from {cache_file}")
                        pbar.update(repeated_times * len(itermax_list))
                        continue
                    else:
                        print(f"Outdated cache detected, recomputing: {cache_file}")

                # Data containers
                time_lp = torch.zeros((repeated_times, len(itermax_list)))
                time_lp_norm = torch.zeros((repeated_times, len(itermax_list)))
                time_rw2 = torch.zeros((repeated_times, len(itermax_list)))
                time_rw2_norm = torch.zeros((repeated_times, len(itermax_list)))

                w2_lp = torch.zeros((repeated_times, len(itermax_list)))
                w2_lp_norm = torch.zeros((repeated_times, len(itermax_list)))
                w2_rw2 = torch.zeros((repeated_times, len(itermax_list)))
                w2_rw2_norm = torch.zeros((repeated_times, len(itermax_list)))

                w2_gt = torch.zeros(repeated_times)

                # Run experiments
                for r in range(repeated_times):
                    X = generate_distribution(source_dist, size, dim)
                    Y = generate_distribution(target_dist, size, dim)

                    # setting (2): translated target
                    Y[:, -1] += 1.0

                    a = torch.ones(size, dtype=X.dtype) / size
                    b = torch.ones(size, dtype=Y.dtype) / size

                    # Ground truth using large itermax
                    w2_gt[r] = exact_lp(X, Y, a, b, itermax=gt_itermax)

                    for i, itmax in enumerate(itermax_list):
                        tic = time.time()
                        w2_lp[r, i] = exact_lp(X, Y, a, b, itermax=itmax)
                        time_lp[r, i] = time.time() - tic

                        tic = time.time()
                        w2_lp_norm[r, i] = exact_lp_norm(X, Y, a, b, itermax=itmax)
                        time_lp_norm[r, i] = time.time() - tic

                        tic = time.time()
                        w2_rw2[r, i] = rw2_lp(X, Y, a, b, itermax=itmax, M=M)
                        time_rw2[r, i] = time.time() - tic

                        tic = time.time()
                        w2_rw2_norm[r, i] = rw2_lp_norm(X, Y, a, b, itermax=itmax, M=M)
                        time_rw2_norm[r, i] = time.time() - tic

                        pbar.update(1)

                res = {
                    "time": {
                        "lp": time_lp.numpy(),
                        "lp_norm": time_lp_norm.numpy(),
                        "rw2": time_rw2.numpy(),
                        "rw2_norm": time_rw2_norm.numpy(),
                    },
                    "w2": {
                        "lp": w2_lp.numpy(),
                        "lp_norm": w2_lp_norm.numpy(),
                        "rw2": w2_rw2.numpy(),
                        "rw2_norm": w2_rw2_norm.numpy(),
                        "gt": w2_gt.numpy(),
                    },
                    "itermax_list": itermax_list,
                    "M": M,
                    "gt_itermax": gt_itermax,
                }

                results[(dim, size)] = res
                with open(cache_file, "wb") as f:
                    pickle.dump(res, f)
                print(f"Saved results to {cache_file}")

    # =======================================================
    # Plot Results
    # =======================================================
    size = sizes[0]

    def plot_runtime():
        plt.figure(figsize=(7.5, 5.5))

        for dim in dims:
            res = results[(dim, size)]
            x = res["itermax_list"]

            mean_lp, std_lp = compute_mean_std(res["time"]["lp"])
            mean_lp_norm, std_lp_norm = compute_mean_std(res["time"]["lp_norm"])
            mean_rw2, std_rw2 = compute_mean_std(res["time"]["rw2"])
            mean_rw2_norm, std_rw2_norm = compute_mean_std(res["time"]["rw2_norm"])

            color_lp = color_map.get(dim, {}).get("lp", "black")
            color_rw2 = color_map.get(dim, {}).get("rw2", "gray")


            plt.errorbar(
                x, mean_lp, yerr=std_lp,
                color=color_lp, linestyle="-", marker="o",
                label=f"LP (dim={dim})"
            )
            plt.errorbar(
                x, mean_lp_norm, yerr=std_lp_norm,
                color=color_lp, linestyle="-.", marker="s",
                label=f"LP + norm (dim={dim})"
            )
            plt.errorbar(
                x, mean_rw2, yerr=std_rw2,
                color=color_rw2, linestyle="-", marker="D",
                label=f"RW2-LP (dim={dim})"
            )
            plt.errorbar(
                x, mean_rw2_norm, yerr=std_rw2_norm,
                color=color_rw2, linestyle="-.", marker="^",
                label=f"RW2-LP + norm (dim={dim})"
            )

        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xlabel("Itermax")
        plt.ylabel("Running time (s)")
        plt.title(f"Runtime vs Itermax | {source_dist} → {target_dist}, size={size}")
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{source_dist}_to_{target_dist}_runtime_vs_itermax.pdf"),
            format="pdf"
        )
        plt.show()

    def plot_error():
        plt.figure(figsize=(7.5, 5.5))

        for dim in dims:
            res = results[(dim, size)]
            x = res["itermax_list"]
            gt = res["w2"]["gt"][:, None]  # shape (repeated_times, 1)

            err_lp = np.abs(res["w2"]["lp"] - gt)
            err_lp_norm = np.abs(res["w2"]["lp_norm"] - gt)
            err_rw2 = np.abs(res["w2"]["rw2"] - gt)
            err_rw2_norm = np.abs(res["w2"]["rw2_norm"] - gt)

            mean_lp, std_lp = compute_mean_std(err_lp)
            mean_lp_norm, std_lp_norm = compute_mean_std(err_lp_norm)
            mean_rw2, std_rw2 = compute_mean_std(err_rw2)
            mean_rw2_norm, std_rw2_norm = compute_mean_std(err_rw2_norm)

            print(f"dim = {dim}")
            print(mean_lp)
            print(mean_lp_norm)

            print(mean_rw2)
            print(mean_rw2_norm)

            color_lp = color_map.get(dim, {}).get("lp", "black")
            color_rw2 = color_map.get(dim, {}).get("rw2", "gray")

            plt.errorbar(
                x, mean_lp, yerr=std_lp,
                color=color_lp, linestyle="-", marker="o",
                label=f"LP (dim={dim})"
            )
            plt.errorbar(
                x, mean_lp_norm, yerr=std_lp_norm,
                color=color_lp, linestyle="-.", marker="s",
                label=f"LP + norm (dim={dim})"
            )
            plt.errorbar(
                x, mean_rw2, yerr=std_rw2,
                color=color_rw2, linestyle="-", marker="D",
                label=f"RW2-LP (dim={dim})"
            )
            plt.errorbar(
                x, mean_rw2_norm, yerr=std_rw2_norm,
                color=color_rw2, linestyle="-.", marker="^",
                label=f"RW2-LP + norm (dim={dim})"
            )

        plt.xscale("log", base=2)
        plt.yscale("log", base=2)
        plt.xlabel("Itermax")
        plt.ylabel(r"Error $|W_2^2 - W_{2,\mathrm{gt}}^2|$")
        plt.title(f"Error vs Itermax | {source_dist} → {target_dist}, size={size}")
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{source_dist}_to_{target_dist}_error_vs_itermax.pdf"),
            format="pdf"
        )
        plt.show()

    plot_runtime()
    plot_error()

    # =======================================================
    # Console summary
    # =======================================================
    print("\n===== Summary (mean runtime and mean error vs ground truth) =====")
    for dim in dims:
        res = results[(dim, size)]
        gt = res["w2"]["gt"][:, None]

        mean_time_lp = res["time"]["lp"].mean()
        mean_time_lp_norm = res["time"]["lp_norm"].mean()
        mean_time_rw2 = res["time"]["rw2"].mean()
        mean_time_rw2_norm = res["time"]["rw2_norm"].mean()

        mean_err_lp = np.abs(res["w2"]["lp"] - gt).mean()
        mean_err_lp_norm = np.abs(res["w2"]["lp_norm"] - gt).mean()
        mean_err_rw2 = np.abs(res["w2"]["rw2"] - gt).mean()
        mean_err_rw2_norm = np.abs(res["w2"]["rw2_norm"] - gt).mean()

        print(
            f"dim={dim:2d} | "
            f"⟨time⟩ LP={mean_time_lp:.4f}s, "
            f"LP+norm={mean_time_lp_norm:.4f}s, "
            f"RW2={mean_time_rw2:.4f}s, "
            f"RW2+norm={mean_time_rw2_norm:.4f}s | "
            f"⟨err⟩ LP={mean_err_lp:.4e}, "
            f"LP+norm={mean_err_lp_norm:.4e}, "
            f"RW2={mean_err_rw2:.4e}, "
            f"RW2+norm={mean_err_rw2_norm:.4e}"
        )


# ===========================================================
# Run when executed
# ===========================================================
if __name__ == "__main__":
    run_experiment(
        source_dist="Gaussian",
        target_dist="Uniform",
        dims=[2, 10],
        sizes=[2048],
        repeated_times=6,
        gt_itermax=1_000_000,
        M=1.0,
    )