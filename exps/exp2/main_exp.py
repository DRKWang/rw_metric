# main_exp.py

from funcs2 import run_experiment, compute_mean_std_by_col, sanitize_config

import matplotlib.pyplot as plt
import numpy as np
import yaml
import os
import matplotlib as mpl


# ===========================================================
# Matplotlib setup
# ===========================================================
mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 14,
    "axes.grid": True,
    "grid.linestyle": "-.",
    "grid.alpha": 0.5,
    "lines.linewidth": 2.0,
    "pdf.fonttype": 42
})


# ===========================================================
# Explicit color mapping
# ===========================================================
color_map = {
    2: {"sinkhorn": "tab:blue", "rw2": "tab:orange"},
    10: {"sinkhorn": "tab:green", "rw2": "tab:red"}
}


# ===========================================================
# Main
# ===========================================================
if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    save_dir = "results_figs"
    os.makedirs(save_dir, exist_ok=True)

    for exp_idx, exp in enumerate(config["experiments"]):

        exp = sanitize_config(exp)

        print(f"\n=== Running experiment {exp_idx+1}/{len(config['experiments'])} ===")

        results = run_experiment(
            exp["source_dist"],
            exp["target_dist"],
            dims=exp["dims"],
            sizes=exp["sizes"],
            t_list=exp["t_list"],
            reg=exp["reg"],
            stopThr=exp["stopThr"],
            max_iter=exp["max_iter"],
            repeated_times=exp["repeated_times"]
        )

        t_list = exp["t_list"]
        dims = exp["dims"]
        size = exp["sizes"][0]

        # remove t=0 for log scale
        t_array = np.array(t_list)
        mask = t_array > 0
        t_plot = t_array[mask]

        # =====================================================
        # FIGURE 1 — Running time
        # =====================================================
        plt.figure(figsize=(8,6))

        for dim in dims:

            key = (exp["source_dist"], exp["target_dist"], dim, size)
            res = results[key]

            sinkhorn_color = color_map.get(dim, {}).get("sinkhorn", "black")
            rw2_color = color_map.get(dim, {}).get("rw2", "gray")

            run_mean_sk, run_std_sk = compute_mean_std_by_col(res["time"]["sk"])
            run_mean_norm, run_std_norm = compute_mean_std_by_col(res["time"]["norm"])
            run_mean_rot, run_std_rot = compute_mean_std_by_col(res["time"]["rot"])
            run_mean_rot_norm, run_std_rot_norm = compute_mean_std_by_col(res["time"]["rot_norm"])

            run_mean_sk = run_mean_sk[mask]
            run_mean_norm = run_mean_norm[mask]
            run_mean_rot = run_mean_rot[mask]
            run_mean_rot_norm = run_mean_rot_norm[mask]

            run_std_sk = run_std_sk[mask]
            run_std_norm = run_std_norm[mask]
            run_std_rot = run_std_rot[mask]
            run_std_rot_norm = run_std_rot_norm[mask]

            # Sinkhorn (no norm)
            plt.errorbar(
                t_plot, run_mean_sk, yerr=run_std_sk,
                color=sinkhorn_color,
                linestyle="-",
                marker="o",
                label=f"Sinkhorn (dim={dim})"
            )

            # Sinkhorn + norm
            plt.errorbar(
                t_plot, run_mean_norm, yerr=run_std_norm,
                color=sinkhorn_color,
                linestyle="-.",
                marker="s",
                label=f"Sinkhorn + norm (dim={dim})"
            )

            # RW2
            plt.errorbar(
                t_plot, run_mean_rot, yerr=run_std_rot,
                color=rw2_color,
                linestyle="-",
                marker="D",
                label=f"RW2 (dim={dim})"
            )

            # RW2 + norm
            plt.errorbar(
                t_plot, run_mean_rot_norm, yerr=run_std_rot_norm,
                color=rw2_color,
                linestyle="-.",
                marker="^",
                label=f"RW2 + norm (dim={dim})"
            )

        plt.xscale("log", base=2)
        # plt.yscale("log", base=2)

        plt.xlabel("Translation $t$")
        plt.ylabel("Running time (s)")
        plt.title(f"{exp['source_dist']} vs {exp['target_dist']} (log₂–linear scale)")

        plt.legend(fontsize=10)
        plt.grid(True, which="both", linestyle="-.", alpha=0.5)
        plt.tight_layout()

        filename_time = f"{exp['source_dist']}_vs_{exp['target_dist']}_time.pdf"
        plt.savefig(os.path.join(save_dir, filename_time))

        print("Saved:", filename_time)

        plt.show()


        # =====================================================
        # FIGURE 2 — Error vs EMD
        # =====================================================
        plt.figure(figsize=(8,6))

        for dim in dims:

            key = (exp["source_dist"], exp["target_dist"], dim, size)
            res = results[key]

            sinkhorn_color = color_map.get(dim, {}).get("sinkhorn", "black")
            rw2_color = color_map.get(dim, {}).get("rw2", "gray")

            err_sk = np.abs(res["w2"]["sk"] - res["w2"]["emd"])
            err_norm = np.abs(res["w2"]["norm"] - res["w2"]["emd"])
            err_rot = np.abs(res["w2"]["rot"] - res["w2"]["emd"])
            err_rot_norm = np.abs(res["w2"]["rot_norm"] - res["w2"]["emd"])

            err_mean_sk, err_std_sk = compute_mean_std_by_col(err_sk)
            err_mean_norm, err_std_norm = compute_mean_std_by_col(err_norm)
            err_mean_rot, err_std_rot = compute_mean_std_by_col(err_rot)
            err_mean_rot_norm, err_std_rot_norm = compute_mean_std_by_col(err_rot_norm)

            err_mean_sk = err_mean_sk[mask]
            err_mean_norm = err_mean_norm[mask]
            err_mean_rot = err_mean_rot[mask]
            err_mean_rot_norm = err_mean_rot_norm[mask]

            err_std_sk = err_std_sk[mask]
            err_std_norm = err_std_norm[mask]
            err_std_rot = err_std_rot[mask]
            err_std_rot_norm = err_std_rot_norm[mask]

            plt.errorbar(
                t_plot, err_mean_sk, yerr=err_std_sk,
                color=sinkhorn_color, linestyle="-", marker="o",
                label=f"Sinkhorn (dim={dim})"
            )

            plt.errorbar(
                t_plot, err_mean_norm, yerr=err_std_norm,
                color=sinkhorn_color, linestyle="-.", marker="s",
                label=f"Sinkhorn + norm (dim={dim})"
            )

            plt.errorbar(
                t_plot, err_mean_rot, yerr=err_std_rot,
                color=rw2_color, linestyle="-", marker="D",
                label=f"RW2 (dim={dim})"
            )

            plt.errorbar(
                t_plot, err_mean_rot_norm, yerr=err_std_rot_norm,
                color=rw2_color, linestyle="-.", marker="^",
                label=f"RW2 + norm (dim={dim})"
            )

        plt.xscale("log", base=2)
        plt.yscale("log", base=2)

        plt.xlabel("Translation $t$")
        plt.ylabel("Absolute error vs EMD")
        plt.title(f"{exp['source_dist']} vs {exp['target_dist']} (log₂–log₂ scale)")

        plt.legend(fontsize=10)
        plt.grid(True, which="both", linestyle="-.", alpha=0.5)
        plt.tight_layout()

        filename_err = f"{exp['source_dist']}_vs_{exp['target_dist']}_error.pdf"
        plt.savefig(os.path.join(save_dir, filename_err))

        print("Saved:", filename_err)

        plt.show()