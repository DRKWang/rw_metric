import torch
import ot
import time
import os
import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm


# ===========================================================
# Matplotlib setup
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
# Color mapping
# ===========================================================
color_map = {
    2: {"lp": "tab:blue", "rw2": "tab:orange"},
    10: {"lp": "tab:green", "rw2": "tab:red"}
}


# ===========================================================
# Distribution generator
# ===========================================================
def generate_distribution(name, n_samples, dim, center=True):

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
        raise ValueError(f"Unknown distribution {name}")

    if center:
        X -= X.mean(dim=0, keepdim=True)

    return X


# ===========================================================
# Cost normalization
# ===========================================================
def normalize_cost(C):
    return C / C.max()


# ===========================================================
# LP solvers
# ===========================================================
def exact_lp(X, Y, a, b, itermax):

    C = torch.cdist(X, Y) ** 2
    return ot.emd2(a, b, C, numItermax=itermax)


def exact_lp_norm(X, Y, a, b, itermax):

    C = torch.cdist(X, Y) ** 2
    Cn = normalize_cost(C)

    return ot.emd2(a, b, Cn, numItermax=itermax) * C.max()


def rw2_lp(X, Y, a, b, itermax):

    mean_X = torch.matmul(X.T, a)
    mean_Y = torch.matmul(Y.T, b)

    Xc = X - mean_X
    Yc = Y - mean_Y

    C = torch.cdist(Xc, Yc) ** 2

    return ot.emd2(a, b, C, numItermax=itermax) + torch.norm(mean_X - mean_Y) ** 2


def rw2_lp_norm(X, Y, a, b, itermax):

    mean_X = torch.matmul(X.T, a)
    mean_Y = torch.matmul(Y.T, b)

    Xc = X - mean_X
    Yc = Y - mean_Y

    C = torch.cdist(Xc, Yc) ** 2
    Cn = normalize_cost(C)

    return ot.emd2(a, b, Cn, numItermax=itermax) * C.max() + torch.norm(mean_X - mean_Y) ** 2


# ===========================================================
# Helper
# ===========================================================
def compute_mean_std_by_col(arr):

    if isinstance(arr, torch.Tensor):
        arr = arr.detach().numpy()

    return arr.mean(axis=0), arr.std(axis=0)


# ===========================================================
# Main experiment
# ===========================================================
def run_experiment(source_dist="Gaussian",
                   dims=[2,10],
                   sizes=[4096],
                   t_list=[1,2,4,8,16],
                   repeated_times=6,
                   itermax=100000,
                   cache_root="results",
                   save_dir="results_figs"):

    os.makedirs(cache_root, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    results = {}

    total_iters = len(dims)*len(sizes)*repeated_times*len(t_list)

    with tqdm(total=total_iters) as pbar:

        for dim in dims:
            for size in sizes:

                cache_file = os.path.join(cache_root,f"{source_dist}_dim{dim}_size{size}.pkl")

                if os.path.exists(cache_file):

                    with open(cache_file,"rb") as f:
                        results[(dim,size)] = pickle.load(f)

                    pbar.update(repeated_times*len(t_list))
                    continue


                time_lp = torch.zeros((repeated_times,len(t_list)))
                time_lp_norm = torch.zeros((repeated_times,len(t_list)))
                time_rw2 = torch.zeros((repeated_times,len(t_list)))
                time_rw2_norm = torch.zeros((repeated_times,len(t_list)))

                w2_lp = torch.zeros((repeated_times,len(t_list)))
                w2_lp_norm = torch.zeros((repeated_times,len(t_list)))
                w2_rw2 = torch.zeros((repeated_times,len(t_list)))
                w2_rw2_norm = torch.zeros((repeated_times,len(t_list)))

                for r in range(repeated_times):

                    for ti,t in enumerate(t_list):

                        X = generate_distribution(source_dist,size,dim)
                        Y = X.clone()

                        Y[:,-1] += t

                        a = torch.ones(size)/size
                        b = torch.ones(size)/size

                        tic = time.time()
                        w2_lp[r,ti] = exact_lp(X,Y,a,b,itermax)
                        time_lp[r,ti] = time.time()-tic

                        tic = time.time()
                        w2_lp_norm[r,ti] = exact_lp_norm(X,Y,a,b,itermax)
                        time_lp_norm[r,ti] = time.time()-tic

                        tic = time.time()
                        w2_rw2[r,ti] = rw2_lp(X,Y,a,b,itermax)
                        time_rw2[r,ti] = time.time()-tic

                        tic = time.time()
                        w2_rw2_norm[r,ti] = rw2_lp_norm(X,Y,a,b,itermax)
                        time_rw2_norm[r,ti] = time.time()-tic

                        pbar.update(1)

                gt_w2 = torch.tensor([t**2 for t in t_list],dtype=torch.float32)

                res = {

                    "time":{
                        "lp":time_lp.numpy(),
                        "lp_norm":time_lp_norm.numpy(),
                        "rw2":time_rw2.numpy(),
                        "rw2_norm":time_rw2_norm.numpy()
                    },

                    "w2":{
                        "lp":w2_lp.numpy(),
                        "lp_norm":w2_lp_norm.numpy(),
                        "rw2":w2_rw2.numpy(),
                        "rw2_norm":w2_rw2_norm.numpy(),
                        "gt":gt_w2.numpy()
                    }
                }

                results[(dim,size)] = res

                with open(cache_file,"wb") as f:
                    pickle.dump(res,f)


    # ===========================================================
    # Plotting
    # ===========================================================

    size = sizes[0]

    def plot_results(title,ylabel,filename,compute_vals):

        plt.figure(figsize=(7.5,5.5))

        for dim in dims:

            res = results[(dim,size)]

            y_lp,e_lp,y_lp_n,e_lp_n,y_rw2,e_rw2,y_rw2_n,e_rw2_n = compute_vals(res)

            color_lp = color_map.get(dim,{}).get("lp","black")
            color_rw2 = color_map.get(dim,{}).get("rw2","gray")

            plt.errorbar(t_list,y_lp,yerr=e_lp,
                         color=color_lp,linestyle="-",marker="o",
                         label=f"LP (dim={dim})")

            plt.errorbar(t_list,y_lp_n,yerr=e_lp_n,
                         color=color_lp,linestyle="-.",marker="s",
                         label=f"LP+norm (dim={dim})")

            plt.errorbar(t_list,y_rw2,yerr=e_rw2,
                         color=color_rw2,linestyle="-",marker="D",
                         label=f"RW2-LP (dim={dim})")

            plt.errorbar(t_list,y_rw2_n,yerr=e_rw2_n,
                         color=color_rw2,linestyle="-.",marker="^",
                         label=f"RW2-LP+norm (dim={dim})")

        plt.xscale("log",base=2)
        # plt.yscale("log",base=2)
        # plt.yticks([2, 3])
        plt.xlabel("Translation magnitude ($t$)")
        plt.ylabel(ylabel)

        plt.title(title)

        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(save_dir,filename),format="pdf")
        plt.show()


    plot_results(
        f"Runtime Comparison | size={size}, itermax={itermax}",
        "Running time (s)",
        f"{source_dist}_runtime.pdf",
        lambda res: (
            *compute_mean_std_by_col(res["time"]["lp"]),
            *compute_mean_std_by_col(res["time"]["lp_norm"]),
            *compute_mean_std_by_col(res["time"]["rw2"]),
            *compute_mean_std_by_col(res["time"]["rw2_norm"]),
        )
    )


    plot_results(
        f"Error vs Ground Truth | size={size}, itermax={itermax}",
        r"Error $|W_2^2 - t^2|$",
        f"{source_dist}_error.pdf",
        lambda res: (
            *compute_mean_std_by_col(np.abs(res["w2"]["lp"]-res["w2"]["gt"])),
            *compute_mean_std_by_col(np.abs(res["w2"]["lp_norm"]-res["w2"]["gt"])),
            *compute_mean_std_by_col(np.abs(res["w2"]["rw2"]-res["w2"]["gt"])),
            *compute_mean_std_by_col(np.abs(res["w2"]["rw2_norm"]-res["w2"]["gt"])),
        )
    )


# ===========================================================
# Run
# ===========================================================
if __name__ == "__main__":

    run_experiment(
        source_dist="Gaussian",
        dims=[2,10],
        sizes=[4096],
        t_list=[1,2,4,8,16],
        repeated_times=6,
        itermax=100000
    )