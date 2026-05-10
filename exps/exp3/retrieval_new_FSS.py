#!/usr/bin/env python
# coding: utf-8

import os
import sys
sys.path.append("../../src")

"""
Thunderstorm Snapshot Retrieval
(Parallel L2 / W / RW / EGW / GW)

Revised version:
- EGW and GW both treat each binary thunderstorm snapshot as a POINT CLOUD
  of active pixels with UNIFORM MASS on active locations only.
- No downsampling for EGW/GW.
- EGW/GW cost matrices are built from pairwise distances between active points,
  not from the full image grid.
- Empty snapshots are skipped for EGW/GW.
- Produces a combined PDF plot across all metrics.
"""

# ============================================================
# Imports
# ============================================================

import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ot

from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from scipy.spatial.distance import euclidean
from scipy.sparse import issparse

from metrics import (
    wasserstein_distance,
    relative_wasserstein_distance,
)

# ============================================================
# Configuration
# ============================================================

CFG = {
    "data_path": "../../data/processed/2014-01-01-2022-12-30_100-10000_bi_mat.pkl",
    "ref_id": 389,   # overwritten below
    "top_k": 5,
    "min_time_diff": 24,
    "enforce_unique_dates": True,
    "metrics": [
        "L2",
        "W1_L2",
        "W2_L2",
        "W4_L2",
        "RW1_L2",
        "RW2_L2",
        "RW4_L2",
        "GW",
        "EGW",
        "FSS",
    ],
    "max_workers": 20,
    "save_dir": "./imgs",
    "cache_dir": "./cache",

    # ---------------- EGW/GW config ----------------
    "egw_epsilon": 5e-2,
    "egw_loss_fun": "square_loss",
    "egw_solver": "PGD",
    "egw_max_iter": 200,
    "egw_tol": 1e-7,
    "egw_cost_metric": "euclidean",
    "egw_normalize_cost": True,
    "egw_skip_empty": True,
}

# ============================================================
# Load dataset
# ============================================================

print("Loading dataset...")
with open(CFG["data_path"], "rb") as f:
    dataset = pickle.load(f)

CFG["ref_id"] = len(dataset) - 1990
reference_matrix, reference_time = dataset[CFG["ref_id"]]

os.makedirs(CFG["save_dir"], exist_ok=True)
os.makedirs(CFG["cache_dir"], exist_ok=True)

REF_CACHE_DIR = os.path.join(CFG["cache_dir"], f"ref_{CFG['ref_id']}")
os.makedirs(REF_CACHE_DIR, exist_ok=True)

# ============================================================
# Utilities
# ============================================================

def ensure_numeric_matrix(M):
    if issparse(M):
        M = M.toarray()
    return np.asarray(M, dtype=float)


def date_only(iso):
    return datetime.fromisoformat(iso).date()


def binary_matrix_to_point_cloud(binary_matrix):
    """
    Convert binary matrix to point cloud of active pixels.

    Returns
    -------
    points : np.ndarray of shape (n, 2)
    weights : np.ndarray of shape (n,)
    """
    M = ensure_numeric_matrix(binary_matrix)
    points = np.argwhere(M == 1)

    if len(points) == 0:
        return np.empty((0, 2), dtype=np.float64), np.empty((0,), dtype=np.float64)

    points = points.astype(np.float64)
    weights = np.ones(len(points), dtype=np.float64) / len(points)
    return points, weights


def point_cloud_cost_matrix(points, metric="euclidean", normalize=True):
    """
    Build intra-space pairwise cost matrix from a point cloud.
    """
    if len(points) == 0:
        return np.empty((0, 0), dtype=np.float64)

    # scale coordinates to [0,1]-ish range for 100x100 grid
    points = points / 100.0

    C = ot.dist(points, points, metric=metric)
    C = np.asarray(C, dtype=np.float64)

    if normalize:
        cmax = np.max(C)
        if cmax > 0:
            C = C / cmax

    return C

from scipy.ndimage import uniform_filter

def fractions_skill_score(A, B, n=9, threshold=0.5):
    """
    Fractions Skill Score (FSS)

    Parameters
    ----------
    A, B : binary matrices
    n : neighborhood size
    """

    A = ensure_numeric_matrix(A)
    B = ensure_numeric_matrix(B)

    A = (A > threshold).astype(float)
    B = (B > threshold).astype(float)

    FA = uniform_filter(A, size=n, mode="constant")
    FB = uniform_filter(B, size=n, mode="constant")

    num = np.sum((FA - FB) ** 2)
    den = np.sum(FA**2 + FB**2)

    if den == 0:
        return 1.0   # null-null case → perfect

    return 1.0 - num / den

# ============================================================
# Precompute reference GW/EGW geometry
# ============================================================

_REF_POINTS, _REF_PROB = binary_matrix_to_point_cloud(reference_matrix)
_REF_COST = point_cloud_cost_matrix(
    _REF_POINTS,
    metric=CFG["egw_cost_metric"],
    normalize=CFG["egw_normalize_cost"],
)

if len(_REF_POINTS) == 0:
    raise ValueError("Reference snapshot has no active pixels; GW/EGW undefined.")

# ============================================================
# Parallel worker globals
# ============================================================

_WORKER_REF = None
_WORKER_REF_TIME = None
_WORKER_METRIC = None
_WORKER_MIN_TIME_DIFF = None

_WORKER_GW_REF_PROB = None
_WORKER_GW_REF_COST = None
_WORKER_GW_EPSILON = None
_WORKER_GW_LOSS_FUN = None
_WORKER_GW_SOLVER = None
_WORKER_GW_MAX_ITER = None
_WORKER_GW_TOL = None
_WORKER_GW_COST_METRIC = None
_WORKER_GW_NORMALIZE_COST = None
_WORKER_GW_SKIP_EMPTY = None


def _init_worker(
    ref,
    ref_time,
    metric,
    min_td,
    gw_ref_prob,
    gw_ref_cost,
    gw_epsilon,
    gw_loss_fun,
    gw_solver,
    gw_max_iter,
    gw_tol,
    gw_cost_metric,
    gw_normalize_cost,
    gw_skip_empty,
):
    global _WORKER_REF, _WORKER_REF_TIME, _WORKER_METRIC, _WORKER_MIN_TIME_DIFF
    global _WORKER_GW_REF_PROB, _WORKER_GW_REF_COST
    global _WORKER_GW_EPSILON, _WORKER_GW_LOSS_FUN, _WORKER_GW_SOLVER
    global _WORKER_GW_MAX_ITER, _WORKER_GW_TOL
    global _WORKER_GW_COST_METRIC, _WORKER_GW_NORMALIZE_COST, _WORKER_GW_SKIP_EMPTY

    _WORKER_REF = ensure_numeric_matrix(ref)
    _WORKER_REF_TIME = datetime.fromisoformat(ref_time)
    _WORKER_METRIC = metric
    _WORKER_MIN_TIME_DIFF = min_td

    _WORKER_GW_REF_PROB = gw_ref_prob
    _WORKER_GW_REF_COST = gw_ref_cost
    _WORKER_GW_EPSILON = gw_epsilon
    _WORKER_GW_LOSS_FUN = gw_loss_fun
    _WORKER_GW_SOLVER = gw_solver
    _WORKER_GW_MAX_ITER = gw_max_iter
    _WORKER_GW_TOL = gw_tol
    _WORKER_GW_COST_METRIC = gw_cost_metric
    _WORKER_GW_NORMALIZE_COST = gw_normalize_cost
    _WORKER_GW_SKIP_EMPTY = gw_skip_empty

# ============================================================
# EGW / GW distances
# ============================================================

def _prepare_query_geometry(M):
    points_q, prob_q = binary_matrix_to_point_cloud(M)

    if len(points_q) == 0:
        if _WORKER_GW_SKIP_EMPTY:
            return None, None
        raise ValueError("Encountered empty snapshot for GW/EGW.")

    cost_q = point_cloud_cost_matrix(
        points_q,
        metric=_WORKER_GW_COST_METRIC,
        normalize=_WORKER_GW_NORMALIZE_COST,
    )
    return prob_q, cost_q


def _egw_distance(M):
    prob_q, cost_q = _prepare_query_geometry(M)
    if prob_q is None:
        return None

    d = ot.gromov.entropic_gromov_wasserstein2(
        _WORKER_GW_REF_COST,
        cost_q,
        _WORKER_GW_REF_PROB,
        prob_q,
        loss_fun=_WORKER_GW_LOSS_FUN,
        epsilon=_WORKER_GW_EPSILON,
        max_iter=_WORKER_GW_MAX_ITER,
        tol=_WORKER_GW_TOL,
        solver=_WORKER_GW_SOLVER,
        verbose=False,
        log=False,
    )
    return float(np.maximum(d, 0.0))


def _gw_distance(M):
    prob_q, cost_q = _prepare_query_geometry(M)
    if prob_q is None:
        return None

    d = ot.gromov.gromov_wasserstein2(
        _WORKER_GW_REF_COST,
        cost_q,
        _WORKER_GW_REF_PROB,
        prob_q,
        loss_fun=_WORKER_GW_LOSS_FUN,
        max_iter=_WORKER_GW_MAX_ITER,
        tol=_WORKER_GW_TOL,
        armijo=False,
        log=False,
    )
    return float(np.maximum(d, 0.0))

# ============================================================
# Worker
# ============================================================

def _worker(args):
    _, M, t = args
    t_dt = datetime.fromisoformat(t)

    if _WORKER_MIN_TIME_DIFF is not None:
        if abs((t_dt - _WORKER_REF_TIME).total_seconds()) < _WORKER_MIN_TIME_DIFF * 3600:
            return None

    M = ensure_numeric_matrix(M)

    if _WORKER_METRIC == "L2":
        d = euclidean(M.flatten(), _WORKER_REF.flatten())

    elif _WORKER_METRIC.startswith("RW"):
        w, l = map(int, _WORKER_METRIC[2:].split("_L"))
        d = relative_wasserstein_distance(M, _WORKER_REF, w_norm=w, l_norm=l)

    elif _WORKER_METRIC.startswith("W"):
        w, l = map(int, _WORKER_METRIC[1:].split("_L"))
        d = wasserstein_distance(M, _WORKER_REF, w_norm=w, l_norm=l)

    elif _WORKER_METRIC == "EGW":
        d = _egw_distance(M)
        if d is None:
            return None

    elif _WORKER_METRIC == "GW":
        d = _gw_distance(M)
        if d is None:
            return None

    elif _WORKER_METRIC == "FSS":
        fss = fractions_skill_score(M, _WORKER_REF, n=9)
        d = 1.0 - fss
    else:
        raise ValueError(f"Unknown metric: {_WORKER_METRIC}")

    return (M, t, float(np.maximum(d, 0.0)))

# ============================================================
# Distance computation + caching + timing
# ============================================================

def compute_or_load_distances(metric):
    """
    Return
    ------
    results : list of (matrix, timestamp, distance)
    elapsed : float
    cache_hit : bool
    """
    cache_path = os.path.join(REF_CACHE_DIR, f"{metric}.pkl")
    t0 = time.perf_counter()

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            results = pickle.load(f)
        elapsed = time.perf_counter() - t0
        print(f"[CACHE HIT] {metric} — {elapsed:.3f}s")
        return results, elapsed, True

    print(f"[CACHE MISS] Computing {metric}")

    tasks = [(i, M, t) for i, (M, t) in enumerate(dataset) if i != CFG["ref_id"]]

    with Pool(
        min(cpu_count(), CFG["max_workers"]),
        initializer=_init_worker,
        initargs=(
            reference_matrix,
            reference_time,
            metric,
            CFG["min_time_diff"],
            _REF_PROB,
            _REF_COST,
            CFG["egw_epsilon"],
            CFG["egw_loss_fun"],
            CFG["egw_solver"],
            CFG["egw_max_iter"],
            CFG["egw_tol"],
            CFG["egw_cost_metric"],
            CFG["egw_normalize_cost"],
            CFG["egw_skip_empty"],
        ),
    ) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(_worker, tasks),
                total=len(tasks),
                desc=metric
            )
        )

    results = [r for r in results if r is not None]
    results.sort(key=lambda x: x[2])

    with open(cache_path, "wb") as f:
        pickle.dump(results, f)

    elapsed = time.perf_counter() - t0
    print(f"[CACHE SAVE] {metric} — {elapsed:.2f}s")
    return results, elapsed, False

# ============================================================
# Top-K selection
# ============================================================

def select_top_k(results, k):
    neighbors, distances = [], []
    used_dates = set()

    for M, t, d in results:
        if CFG["enforce_unique_dates"]:
            d0 = date_only(t)
            if d0 in used_dates:
                continue
            used_dates.add(d0)

        neighbors.append((M, t))
        distances.append(d)

        if len(neighbors) >= k:
            break

    return neighbors, distances

# ============================================================
# Plot combined PDF
# ============================================================

def plot_combined_metrics_pdf(
    reference,
    reference_time,
    results_by_metric,
    timing_info,
    top_k,
    save_pdf_path
):
    metrics = list(results_by_metric.keys())
    metric_labels = {
        "L2": r"$\ell_2$",
        "W1_L2": r"$W_1$",
        "W2_L2": r"$W_2$",
        "W4_L2": r"$W_4$",

        "RW1_L2": r"$RW_1$",
        "RW2_L2": r"$RW_2$",
        "RW4_L2": r"$RW_4$",
        "GW": r"$GW$",
        "EGW": r"$EGW$",
        "FSS": r"$FSS$",
    }

    n_rows = len(metrics)
    n_cols = top_k + 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.8 * n_cols, 2.4 * n_rows),
        squeeze=False
    )

    ref_mat = ensure_numeric_matrix(reference)
    ref_time_str = datetime.fromisoformat(reference_time).strftime("%Y-%m-%d %H:%M")

    for i, metric in enumerate(metrics):
        row = results_by_metric[metric]
        t_sec = timing_info[metric]["time"]

        ax = axes[i, 0]
        ax.imshow(ref_mat, cmap="viridis")
        ax.set_title(f"Reference\n{ref_time_str}", fontsize=13, pad=2)
        ax.set_ylabel(f"({t_sec:.2f}s)", fontsize=11, labelpad=10)
        ax.axis("off")

        ax.text(
            -0.2, 0.5,
            metric_labels.get(metric, metric),
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="center",
            ha="right"
        )

        for j, ((m, t), d) in enumerate(zip(row["neighbors"], row["distances"]), start=1):
            ax = axes[i, j]
            ax.imshow(ensure_numeric_matrix(m), cmap="viridis")
            ax.set_title(
                f"{datetime.fromisoformat(t).strftime('%Y-%m-%d %H:%M')}\n"
                rf"$d={d:.3f}$",
                fontsize=13,
                pad=2
            )
            ax.axis("off")

    fig.subplots_adjust(
        left=0.07,
        right=0.995,
        top=0.99,
        bottom=0.01,
        wspace=0.01,
        hspace=0.25
    )

    plt.savefig(save_pdf_path, format="pdf", bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"[Saved PDF] {save_pdf_path}")

# ============================================================
# Main experiment
# ============================================================

combined_results = {}
timing_info = {}

print("\nReference timestamp:", reference_time)
print("Reference matrix shape:", ensure_numeric_matrix(reference_matrix).shape)
print("Reference GW/EGW active points:", len(_REF_POINTS))
print("EGW epsilon:", CFG["egw_epsilon"])
print("EGW solver:", CFG["egw_solver"])

for metric in CFG["metrics"]:
    print("\n" + "=" * 60)
    print(f"Metric: {metric}")

    all_results, elapsed, cache_hit = compute_or_load_distances(metric)
    neighbors, distances = select_top_k(all_results, CFG["top_k"])

    combined_results[metric] = {
        "neighbors": neighbors,
        "distances": distances
    }

    timing_info[metric] = {
        "time": elapsed,
        "cache_hit": cache_hit
    }

    df = pd.DataFrame({
        "rank": np.arange(1, len(neighbors) + 1),
        "timestamp": [t for _, t in neighbors],
        "date": [date_only(t) for _, t in neighbors],
        "distance": distances
    })
    print(df)

# ============================================================
# Save timing summary
# ============================================================

timing_df = pd.DataFrame.from_dict(timing_info, orient="index")
timing_df.index.name = "metric"

timing_csv_path = os.path.join(CFG["save_dir"], "metric_timing.csv")
timing_df.to_csv(timing_csv_path)

print("\nTiming summary:")
print(timing_df)

# ============================================================
# Combined PDF
# ============================================================

combined_pdf = os.path.join(
    CFG["save_dir"],
    f"ref{CFG['ref_id']}_all_metrics_with_gw.pdf"
)

plot_combined_metrics_pdf(
    reference_matrix,
    reference_time,
    combined_results,
    timing_info,
    CFG["top_k"],
    combined_pdf
)

print("\nExperiment completed.")