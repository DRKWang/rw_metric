from scipy.spatial.distance import jensenshannon
from tqdm import tqdm

import numpy as np
from scipy.spatial.distance import euclidean
from scipy.sparse import csr_matrix
from datetime import datetime


def normalize_vector(matrix):
    """
    Flattens the matrix into a vector and normalizes it to create a probability distribution.
    """
    vector = matrix.flatten()
    return vector / vector.sum()


def js_divergence(matrix1, matrix2):
    """
    js_div
    Computes the Jensen-Shannon divergence between two matrices.
    """
    # Normalize the matrices to form valid probability distributions
    prob_dist1 = normalize_vector(matrix1)
    prob_dist2 = normalize_vector(matrix2)

    # Compute the J-S divergence using scipy
    js_div = jensenshannon(prob_dist1, prob_dist2) ** 2
    return js_div

import ot  # POT: Python Optimal Transport

def binary_matrix_to_point_cloud(binary_matrix):
    """
    Converts a binary matrix into a point cloud.

    Args:
    - binary_matrix (np.ndarray): A binary matrix (values 0 or 1).

    Returns:
    - np.ndarray: Point cloud as a 2D array of coordinates [(row, col), ...].
    - np.ndarray: Weights of each point (uniform weights for binary matrices).
    """
    # Get the indices where the matrix value is 1
    points = np.argwhere(binary_matrix == 1)
    # Uniform weights for each point
    weights = np.ones(len(points)) / len(points)
    return points, weights

def wasserstein_distance(
    matrix1,
    matrix2,
    w_norm=2,
    l_norm=2,
    normalize=True
):
    """
    Computes the Wasserstein distance between two binary matrices using POT.

    Normalization rescales coordinates to [0,1]^2 to make the distance
    independent of grid resolution.
    """

    if w_norm < 1:
        raise ValueError("Invalid w_norm! Must be >= 1.")
    if l_norm < 1:
        raise ValueError("Invalid l_norm! Must be >= 1.")

    # Convert binary matrices to point clouds and weights
    x, a = binary_matrix_to_point_cloud(matrix1)
    y, b = binary_matrix_to_point_cloud(matrix2)

    # -------------------------------------------------
    # NEW: coordinate normalization
    # -------------------------------------------------
    if normalize:
        scale = np.array(matrix1.shape, dtype=float)
        x = x / scale
        y = y / scale

    # Compute cost matrix
    C = ot.dist(x, y, metric="minkowski", p=l_norm)

    # Wasserstein p-cost
    C = C ** w_norm

    # Solve OT
    Wp = ot.emd2(a, b, C)

    return Wp ** (1.0 / w_norm)

# def relative_wasserstein_distance(matrix1, matrix2, w_norm=2, l_norm=2,
#                                   tau=1e-2, max_iter=100, eta=1e-1,
#                                   inner_max_iter=50, inner_tol=1e-6,
#                                   normalize=True, verbose=True):
#     """
#     Computes the Relative Wasserstein (RW_p) distance between two binary matrices.
#
#     RW_p = min_t min_P sum_{i,j} ||x_i + t - y_j||^p P_ij
#            s.t. P >= 0,  P 1 = a,  P^T 1 = b.
#
#     This metric is translation-invariant and often more robust to positional shifts
#     than the standard Wasserstein distance.
#
#     Args:
#     -------
#     matrix1 : np.ndarray
#         First binary (or grayscale) matrix.
#     matrix2 : np.ndarray
#         Second binary (or grayscale) matrix.
#     w_norm : float
#         Exponent p in RW_p (must be >= 1).
#     l_norm : float
#         Exponent for Minkowski cost norm (currently unused but kept for consistency).
#     tau : float
#         Convergence tolerance for outer loop.
#     max_iter : int
#         Maximum number of alternating updates (P, t).
#     eta : float
#         Learning rate for inner gradient updates on t.
#     inner_max_iter : int
#         Maximum inner-loop iterations for t-step.
#     inner_tol : float
#         Inner-loop convergence tolerance.
#     normalize : bool, default=True
#         If True, normalize spatial coordinates to [0, 1] range for scale-invariant distance.
#     verbose : bool
#         If True, print optimization progress per iteration.
#
#     Returns:
#     --------
#     float
#         The RW_p distance between the two matrices.
#     """
#     # Convert binary matrices to point clouds and weights
#     x, a = binary_matrix_to_point_cloud(matrix1)
#     y, b = binary_matrix_to_point_cloud(matrix2)
#
#     # === Optional normalization ===
#     if normalize:
#         x = x / np.array(matrix1.shape)
#         y = y / np.array(matrix2.shape)
#
#     # n, d = x.shape
#     # m = y.shape[0]
#
#     # === Initialization ===
#     t = np.mean(y, axis=0) - np.mean(x, axis=0)
#     F_prev = np.inf
#
#     # === Alternating optimization ===
#     for k in range(max_iter):
#         # (P-step): Optimal coupling for fixed t
#         C = np.linalg.norm(x[:, None, :] + t[None, None, :] - y[None, :, :], axis=2) ** w_norm
#         if k == 0:
#             res = ot.solve(C, a, b, verbose=False)
#         else:
#             res = ot.solve(C, a, b, verbose=False, potentials_init=(dual_a, dual_b))
#         P_new = res.plan
#         dual_a, dual_b = res.potentials
#
#         # (t-step): Gradient-based update
#         t_inner = t.copy()
#         for it in range(inner_max_iter):
#             diff = x[:, None, :] + t_inner[None, None, :] - y[None, :, :]
#             grad = w_norm * np.sum(
#                 P_new[:, :, None]
#                 * (np.linalg.norm(diff, axis=2) ** (w_norm - 2))[:, :, None]
#                 * diff,
#                 axis=(0, 1)
#             )
#             t_next = t_inner - eta * grad
#
#             # Check inner convergence
#             if np.linalg.norm(t_next - t_inner) < inner_tol * max(1.0, np.linalg.norm(t_inner)):
#                 t_inner = t_next
#                 break
#             t_inner = t_next
#
#         t_new = t_inner
#
#         # (Eval) Objective value
#         F_new = np.sum(
#             P_new * (np.linalg.norm(x[:, None, :] + t_new[None, None, :] - y[None, :, :], axis=2) ** w_norm)
#         )
#
#         # Convergence check
#         rel_change = np.abs(F_prev - F_new) / max(1e-12, np.abs(F_prev))
#         if verbose:
#             print(f"[{k:03d}] F = {F_new:.6f}, ΔF/F = {rel_change:.3e}, |t| = {np.linalg.norm(t_new):.3e}")
#         if rel_change < tau:
#             break
#
#         # Prepare for next iteration
#         F_prev, P, t = F_new, P_new, t_new
#
#     # Return RW_p distance in same scale as Wasserstein
#     return F_new ** (1 / w_norm)
#
#


def relative_wasserstein_distance(matrix1, matrix2, w_norm=2, l_norm=2,
                                  tau=1e-2, max_iter=100,
                                  inner_max_iter=20, inner_tol=1e-8,
                                  normalize=True, verbose=False,
                                  armijo_beta=0.5, armijo_sigma=1e-4,
                                  armijo_max_backtracks=20):
    """
    RW_p with Armijo line search in the t-step.
    - For p=2: closed-form t (single OT solve).
    - For p!=2: alternating (P,t) with Armijo backtracking for fast inner convergence.
    """

    # --- helpers (unchanged) ---
    def binary_matrix_to_point_cloud(M):
        coords = np.argwhere(M > 0)
        if coords.size == 0:
            coords = np.zeros((1, M.ndim))
            w = np.array([0.0])
        else:
            w_raw = M[tuple(coords.T)].astype(float)
            w = w_raw / np.sum(w_raw)
        return coords.astype(float), w

    # Convert to weighted point clouds
    x, a = binary_matrix_to_point_cloud(matrix1)
    y, b = binary_matrix_to_point_cloud(matrix2)
    if np.sum(a) == 0 or np.sum(b) == 0:
        return 10000.0

    if normalize:
        scale = np.array(matrix1.shape, dtype=float)
        x = x / scale
        y = y / scale

    # p = 2: closed-form t*, single OT solve
    if abs(w_norm - 2.0) < 1e-12:
        t_star = np.sum(b[:, None] * y, axis=0) - np.sum(a[:, None] * x, axis=0)
        C = np.linalg.norm(x[:, None, :] + t_star[None, None, :] - y[None, :, :], axis=2) ** 2
        res = ot.solve(C, a, b, verbose=False)
        F = float(np.sum(res.plan * C))
        if verbose:
            print(f"[p=2] single-solve: F={F:.6e}, |t*|={np.linalg.norm(t_star):.3e}")
        return F ** 0.5

    # p != 2: alternating with Armijo on t
    t = np.sum(b[:, None] * y, axis=0) - np.sum(a[:, None] * x, axis=0)
    F_prev = np.inf
    dual_a = dual_b = None

    for k in range(max_iter):
        # --- P-step ---
        diff = x[:, None, :] + t[None, None, :] - y[None, :, :]
        dists = np.linalg.norm(diff, axis=2)
        C = dists ** w_norm
        if dual_a is None:
            res = ot.solve(C, a, b, verbose=False)
        else:
            res = ot.solve(C, a, b, verbose=False, potentials_init=(dual_a, dual_b))
        P = res.plan
        dual_a, dual_b = res.potentials

        # --- t-step: Armijo backtracking along -grad ---
        eps = 1e-12

        def obj_and_grad(t_vec):
            D = x[:, None, :] + t_vec[None, None, :] - y[None, :, :]
            r = np.linalg.norm(D, axis=2)
            F = float(np.sum(P * (r ** w_norm)))
            if w_norm == 1:
                # subgradient with safe smoothing
                G = np.sum(P[:, :, None] * (D / np.maximum(r[:, :, None], eps)), axis=(0, 1))
            else:
                G = w_norm * np.sum(P[:, :, None] * (r ** (w_norm - 2))[:, :, None] * D, axis=(0, 1))
            return F, G

        t_inner = t.copy()
        F_curr, g_curr = obj_and_grad(t_inner)
        gnorm = float(np.linalg.norm(g_curr))

        if gnorm > 0:
            # scale-aware initial step; try to grow if successful
            step = 1.0 / max(1.0, gnorm)
            for it in range(inner_max_iter):
                # Backtracking until Armijo condition is met
                bt = 0
                while bt < armijo_max_backtracks:
                    t_trial = t_inner - step * g_curr
                    F_trial, _ = obj_and_grad(t_trial)
                    if F_trial <= F_curr - armijo_sigma * step * (gnorm ** 2):
                        break
                    step *= armijo_beta
                    bt += 1

                # Accept step
                t_next = t_trial
                if np.linalg.norm(t_next - t_inner) < inner_tol * max(1.0, np.linalg.norm(t_inner)):
                    t_inner = t_next
                    F_curr = F_trial
                    break

                # Prepare next inner iteration (try mild step growth)
                t_inner = t_next
                F_curr, g_curr = obj_and_grad(t_inner)
                gnorm = float(np.linalg.norm(g_curr))
                if gnorm < 1e-14:
                    break
                step *= 1.2  # modest growth after a successful step

        t_new = t_inner

        # --- evaluate and stopping for outer loop ---
        diff = x[:, None, :] + t_new[None, None, :] - y[None, :, :]
        F_new = float(np.sum(P * (np.linalg.norm(diff, axis=2) ** w_norm)))

        rel_change = np.abs(F_prev - F_new) / max(1e-12, np.abs(F_prev))
        if verbose:
            print(f"[{k:03d}] F = {F_new:.6e}, ΔF/F = {rel_change:.3e}, |t| = {np.linalg.norm(t_new):.3e}")
        t = t_new
        if rel_change < tau:
            F_prev = F_new
            break
        F_prev = F_new

    return F_prev ** (1.0 / w_norm)

def find_closest_matrices(dataset, reference_matrix, num=1, metric='euclidean', min_time_diff=None):
    """
    Finds the closest matrices in the dataset to the reference matrix based on the given metric,
    while ensuring the time difference between selected matrices exceeds a specified value in hours.

    Args:
    - dataset (list of tuples): List of (matrix, time_label) pairs. Matrices can be NumPy arrays or sparse matrices.
    - reference_matrix (np.ndarray or csr_matrix): The reference matrix to compare against.
    - num (int): The desired number of final output matrices.
    - metric (str): The metric to use for comparison ('euclidean', 'cosine', or Wasserstein variants).
    - min_time_diff (float, optional): Minimum time difference in hours between output matrices' time labels.

    Returns:
    - tuple:
        - subset (list): The closest elements as [(matrix, label), ...].
        - distances (list): The distances of the closest matrices.
    """
    # Convert reference matrix to a dense array if it is sparse
    if isinstance(reference_matrix, csr_matrix):
        reference_matrix = reference_matrix.toarray()

    # Sort the dataset by time (ascending)
    dataset.sort(key=lambda x: datetime.fromisoformat(x[1]))

    # List to store distances and corresponding matrices
    distances = []

    for matrix, time_label in tqdm(dataset):
        # Convert matrix to dense if it is sparse
        if isinstance(matrix, csr_matrix):
            matrix = matrix.toarray()

        # Flatten matrices for comparison (if 2D or higher-dimensional)
        flat_matrix = matrix.flatten()
        flat_reference = reference_matrix.flatten()

        # Compute distance based on the selected metric
        if metric == 'euclidean':
            distance = euclidean(flat_matrix, flat_reference)
        elif metric == 'js_div':
            distance = js_divergence(flat_matrix, flat_reference)
        elif metric.startswith('RW'):
            # Parse RW parameters, e.g. "RW2_L2"
            try:
                parts = metric[2:].split('_L')
                w_norm = int(parts[0])
                l_norm = int(parts[1])
                distance = relative_wasserstein_distance(matrix, reference_matrix,
                                                         w_norm=w_norm, l_norm=l_norm)
            except (IndexError, ValueError):
                raise ValueError("Invalid Relative Wasserstein metric format. Use 'RW<w_norm>_L<l_norm>'.")
        elif metric.startswith('W'):
            # Parse Wasserstein parameters from metric string, e.g., 'W1_L2'
            try:
                parts = metric[1:].split('_L')
                w_norm = int(parts[0])
                l_norm = int(parts[1])
                distance = wasserstein_distance(matrix, reference_matrix,
                                                w_norm=w_norm, l_norm=l_norm)
            except (IndexError, ValueError):
                raise ValueError("Invalid Wasserstein metric format. Use 'W<w_norm>_L<l_norm>'.")

        else:
            raise ValueError("Unsupported metric. Use 'euclidean', 'js_div', or Wasserstein variants.")

        # Append matrix, label, and distance to the list
        distances.append((matrix, time_label, distance))

    # Sort by distance
    distances.sort(key=lambda x: x[2])

    # Extract the closest matrices, enforcing the minimum time difference
    closest_subset = []
    closest_distances = []
    selected_times = []

    for matrix, label, distance in distances:
        time_dt = datetime.fromisoformat(label)

        # Enforce minimum time difference in hours
        if min_time_diff is not None:
            min_time_diff_seconds = min_time_diff * 3600  # Convert hours to seconds
            if any(abs((time_dt - selected_time).total_seconds()) < min_time_diff_seconds for selected_time in selected_times):
                continue

        # Add to results
        closest_subset.append((matrix, label))
        closest_distances.append(distance)
        selected_times.append(time_dt)

        # Stop if we have exactly `num` outputs
        if len(closest_subset) >= num:
            break

    return closest_subset, closest_distances


# Example matrices
if __name__ == "__main__":
    matrix1 = np.array([[1, 2, 3], [4, 5, 6]])
    matrix2 = np.array([[7, 8, 9], [10, 11, 12]])

    # Compute the J-S divergence
    jsd = js_divergence(matrix1, matrix2)
    print(f"Jensen-Shannon Divergence: {jsd}")
    # Define the values
    M1 = 1.5 / 2
    M2 = 0.5 / 2

    # Compute the given expression
    # Compute the modified expression
    result_modified = 1 * np.log(1 / M1) + 0.5 * np.log(0.5 / M1) + 0.5 * np.log(0.5 / M2)
    np.sqrt(result_modified / 2)


