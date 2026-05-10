# Relative Translation Invariant Wasserstein Distance (RWp)

This repository accompanies the paper **["Relative Translation Invariant Wasserstein Distance"](docs/TMLR_RWp_metric_final.pdf)**.
It introduces a family of optimal-transport-based distances, denoted **RWp**, for comparing probability distributions while ignoring global relative translations. The method is designed to capture intrinsic structural differences between distributions rather than differences caused by translations in location.

## Overview

Classical Wasserstein distances compare probability distributions within a fixed coordinate system. When two distributions have similar shapes but are shifted far apart, the classical Wasserstein distance can be dominated by the translation between their centers, causing the intrinsic difference between the distributions to be overlooked.

This project addresses this limitation by introducing **relative translation optimal transport (ROT)** and the corresponding **relative translation invariant Wasserstein distance**.

The proposed distance is useful when the important signal is the internal geometry or morphology of a distribution rather than its absolute position. Applications explored in the paper include numerical stabilization of optimal transport solvers and retrieval of similar thunderstorm patterns from large-scale radar data.

## Key Concepts

### Relative Translation Optimal Transport

Given two probability distributions `mu` and `nu`, the ROT problem searches for the best translation vector `t` that aligns one distribution with the other before computing optimal transport:

```text
ROT(mu, nu, p) = inf_t OT(mu + t, nu, p)
```

Here, `mu + t` denotes the distribution obtained by translating `mu` by vector `t`.

### Relative Translation Invariant Wasserstein Distance

The RWp distance is defined as:

```text
RWp([mu], [nu]) = ROT(mu, nu, p)^(1/p)
```

where `[mu]` and `[nu]` are equivalence classes of distributions under translation. This means that distributions differing only by a global translation are treated as equivalent.

The paper proves that **RWp is a valid metric** on the quotient space of probability distributions modulo translations.

## Main Contributions

- Introduces a new family of distances, **relative translation invariant Wasserstein distances** `RWp`, for `p >= 1`.
- Proves that `RWp` defines a true metric on the quotient space induced by translation equivalence.
- Develops a bi-level alternating algorithm for computing general `RWp` distances between discrete distributions.
- Shows that, for `p = 2`, the optimal coupling matrix is invariant under relative translations.
- Proposes two numerically stable algorithms:
  - `RW2-LP`
  - `RW2-Sinkhorn`
- Demonstrates practical use in large-scale thunderstorm pattern retrieval.

## Algorithms

### General RWp Algorithm

For arbitrary `p >= 1`, the ROT problem is solved with an alternating optimization scheme:

1. Fix the translation vector `t` and solve a standard discrete optimal transport problem for the coupling matrix `P`.
2. Fix the coupling matrix `P` and update `t` by minimizing a convex objective.
3. Repeat until convergence.

The paper uses warm-started LP solves, dual-simplex reinitialization, and Armijo backtracking to improve efficiency and stability.

### RW2-LP Algorithm

For `p = 2`, the squared Wasserstein distance decomposes into a mean-shift component and a centered distribution component:

```text
W2^2(mu, nu) = ||mean(mu) - mean(nu)||^2 + RW2^2([mu], [nu])
```

The `RW2-LP` algorithm uses this decomposition to translate one distribution by the difference of the means before solving the optimal transport LP. This can reduce the magnitude of the cost matrix and improve numerical stability.

### RW2-Sinkhorn Algorithm

The `RW2-Sinkhorn` algorithm applies the same mean-alignment idea to entropy-regularized optimal transport. The method can reduce underflow in the Gibbs kernel:

```text
K_ij = exp(-C_ij / lambda)
```

The paper also proves that the Sinkhorn convergence rate in Hilbert's projective metric is invariant under translation, so the stabilization does not change the theoretical convergence rate.

## Experiments

The paper evaluates the proposed methods in three main settings.

### 1. Numerical Validation of RW2-LP

The `RW2-LP` method is compared with standard LP-based optimal transport solvers on translated distributions. Results show that mean alignment significantly reduces numerical error, especially when the translation magnitude is large or the dimension is higher.

### 2. Numerical Validation of RW2-Sinkhorn

The `RW2-Sinkhorn` method is tested on several source-target distribution pairs, including Gaussian-to-uniform, Gaussian-to-Gaussian, Gaussian-to-geometric, and Gaussian-to-Poisson settings. Results show improved numerical stability compared to standard Sinkhorn, particularly under large translations.

### 3. Thunderstorm Pattern Retrieval

The method is applied to radar-based thunderstorm pattern retrieval near Dallas Fort Worth International Airport. Radar snapshots are converted into binary distributions, and `RWp` distances are used to retrieve visually and structurally similar storm patterns.

The experiments compare `RWp` against several baselines, including:

- Euclidean distance
- Fractions Skill Score (FSS)
- Classical Wasserstein distance `W2`
- Centered Wasserstein distances
- Gromov-Wasserstein distance
- Entropic Gromov-Wasserstein distance

The results show that `RWp` distances better capture storm morphology and orientation while being robust to global translation.

## Recommended Use Cases

RWp distances are especially useful when comparing distributions where shape matters more than absolute position, such as:

- Weather and radar pattern retrieval
- Image or point-cloud matching under translation
- Shape-based distribution comparison
- Robust optimal transport under systematic coordinate shifts
- Numerical stabilization of optimal transport solvers

## Repository Structure

A suggested repository layout is:

```text
rw_metrics/
├── docs/
│   ├── tmlr_rwp_Metric.pdf/
├── exps/
│   ├── exp0_plot_WpL/
│   ├── exp1/
│   ├── exp2/
│   └── exp3/
├── src/
│   └── metrics.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```


Adjust this structure to match the actual implementation.

## Installation

Create a Python environment and install the required packages:

```bash
git clone git@github.com:DRKWang/rw_metric.git
cd rw_metric

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Common dependencies may include:

```text
matplotlib
ipywidgets
ipympl
ipykernel
numpy<2
tqdm
pot
scikit-learn
pandas
jupyter
notebook
sklearn_extra
torch
scipy
```

The Python Optimal Transport package, commonly imported as `ot`, is used for LP-based and Sinkhorn optimal transport computations.

## Citation

If you use this work, please cite:

```bibtex
@article{wang2026relative,
  title={Relative Translation Invariant Wasserstein Distance},
  author={Wang, Binshuai and Di, Qiwei and Yin, Ming and Wang, Mengdi and Gu, Quanquan and Wei, Peng},
  year={2026}
}
```

## License

This repository is licensed under the GNU General Public License v3.0.

```text
GNU General Public License v3.0
```

See the `LICENSE` file for the full license text.
