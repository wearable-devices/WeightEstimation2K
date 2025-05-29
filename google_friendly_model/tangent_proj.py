
import numpy as np


def riemann_mean_cmsis(matrices, max_iter=10, epsilon=1e-6):
    """
    Specialized implementation of Riemannian mean calculation for 3x3 SPD matrices
    using CMSIS-DSP matrix and vector operations.
    """
    n_matrices = matrices.shape[0]

    # Scale the input matrices to prevent overflow using CMSIS-DSP
    max_val = 0
    for mat in matrices:
        flat_mat = mat.ravel()
        abs_vec = dsp.arm_abs_f32(flat_mat)
        curr_max = dsp.arm_max_f32(abs_vec)[0]
        max_val = max(max_val, curr_max)

    scale_factor = 1.0
    if max_val > 1e10:
        scale_factor = 1e10 / max_val
        matrices = dsp.arm_mat_scale_f32(matrices, scale_factor)[1]

    # Initialize with arithmetic mean using CMSIS-DSP
    mean_matrix = np.zeros((3, 3), dtype=np.float32)
    for i in range(n_matrices):
        mean_matrix = dsp.arm_mat_add_f32(mean_matrix, matrices[i])[1]
    mean_matrix = dsp.arm_mat_scale_f32(mean_matrix, 1.0 / n_matrices)[1]

    # Ensure symmetry using CMSIS-DSP
    mean_matrix_t = dsp.arm_mat_trans_f32(mean_matrix)[1]
    sum_matrix = dsp.arm_mat_add_f32(mean_matrix, mean_matrix_t)[1]
    mean_matrix = dsp.arm_mat_scale_f32(sum_matrix, 0.5)[1]

    # Add convergence history tracking
    convergence_history = []

    # Iterative procedure for 3x3 matrices
    for n_iter in range(max_iter):
        # Use C-based SVD
        U, S, V = svd3(mean_matrix)

        # For symmetric matrices, eigenvalues are singular values
        # We need to determine their signs by checking U·V
        signs = np.diag(np.sign(U.T @ V))
        eigvals = np.diag(S) * signs

        # Ensure positive eigenvalues for sqrt and log
        eigvals = np.maximum(dsp.arm_abs_f32(eigvals), 1e-10)

        # Ensure eigenvectors are orthonormal
        eigvecs = U

        # Compute mean^(-1/2) using CMSIS-DSP operations
        isqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
        temp = dsp.arm_mat_mult_f32(eigvecs, isqrt_diag)[1]
        mean_isqrt = dsp.arm_mat_mult_f32(temp, dsp.arm_mat_trans_f32(eigvecs)[1])[1]

        # Storage for sum of logarithms
        log_sum = np.zeros((3, 3), dtype=np.float32)

        for i in range(n_matrices):
            # Compute whitened matrix using CMSIS-DSP
            temp1 = dsp.arm_mat_mult_f32(mean_isqrt, matrices[i])[1]
            whitened = dsp.arm_mat_mult_f32(temp1, mean_isqrt)[1]

            # Ensure symmetry using CMSIS-DSP
            whitened_transposed = np.zeros_like(whitened)
            whitened_transposed = dsp.arm_mat_trans_f32(whitened)[1]
            whitened = dsp.arm_mat_add_f32(whitened, whitened_transposed)[1]
            whitened = dsp.arm_mat_scale_f32(whitened, 0.5)[1]

            # Use SVD for whitened matrix
            Uw, Sw, Vw = svd3(whitened)
            signs_w = np.diag(np.sign(dsp.arm_mat_trans_f32(Uw)[1] @ Vw))
            eigvals_w = np.diag(Sw) * signs_w

            # Ensure positive values for log
            eigvals_w = np.maximum(dsp.arm_abs_f32(eigvals_w), 1e-10)

            # Ensure orthonormal eigenvectors
            eigvecs_w = Uw

            # Compute matrix logarithm using CMSIS-DSP
            log_diag = np.diag(np.log(eigvals_w)).astype(np.float32)
            temp2 = dsp.arm_mat_mult_f32(eigvecs_w, log_diag)[1]
            log_whitened = dsp.arm_mat_mult_f32(temp2, dsp.arm_mat_trans_f32(eigvecs_w)[1])[1]

            # Add to sum using CMSIS-DSP
            log_sum = dsp.arm_mat_add_f32(log_sum, log_whitened)[1]

        # Compute mean tangent vector using CMSIS-DSP
        mean_tangent = dsp.arm_mat_scale_f32(log_sum, 1.0 / n_matrices)[1]

        # Check for convergence using Frobenius norm
        # Compute using CMSIS-DSP dot product for each row
        norm_sq = 0.0
        for i in range(3):
            norm_sq += dsp.arm_dot_prod_f32(mean_tangent[i], mean_tangent[i])
        norm = np.sqrt(norm_sq)
        convergence_history.append(norm)

        if norm < epsilon:
            break

        # Update mean estimate using matrix exponential
        Ut, St, Vt = svd3(mean_tangent.astype(np.float32))
        signs_t = np.diag(np.sign(Ut.T @ Vt))
        eigvals_t = np.diag(St) * signs_t
        eigvecs_t = Ut

        # Compute matrix exponential using CMSIS-DSP
        exp_diag = np.diag(np.exp(eigvals_t)).astype(np.float32)
        temp3 = dsp.arm_mat_mult_f32(eigvecs_t, exp_diag)[1]
        mean_update = dsp.arm_mat_mult_f32(temp3, dsp.arm_mat_trans_f32(eigvecs_t)[1])[1]

        # Update mean using current eigendecomposition
        sqrt_diag = np.diag(np.sqrt(eigvals)).astype(np.float32)
        temp4 = dsp.arm_mat_mult_f32(eigvecs, sqrt_diag)[1]
        mean_sqrt = dsp.arm_mat_mult_f32(temp4, dsp.arm_mat_trans_f32(eigvecs)[1])[1]

        # Final update using CMSIS-DSP
        temp5 = dsp.arm_mat_mult_f32(mean_sqrt, mean_update)[1]
        mean_matrix = dsp.arm_mat_mult_f32(temp5, mean_sqrt)[1]

        # Ensure symmetry after update
        mean_matrix_t = dsp.arm_mat_trans_f32(mean_matrix)[1]
        sum_matrix = dsp.arm_mat_add_f32(mean_matrix, mean_matrix_t)[1]
        mean_matrix = dsp.arm_mat_scale_f32(sum_matrix, 0.5)[1]

    # Unscale the result
    if scale_factor != 1.0:
        mean_matrix = mean_matrix / scale_factor

    return mean_matrix, convergence_history

def tangent_space_relative_to_mean_cmsis(spd_matrices):
    # Compute Riemannian mean using CMSIS-DSP implementation
    P_cmsis, _ = riemann_mean_cmsis(spd_matrices)

    # Compute tangent vectors using all three methods
    n_matrices = np.shape(spd_matrices)[0]

    # 6 unique elements for 3x3 matrix
    tangent_vectors_cmsis = np.zeros((n_matrices, 6))
    for i in range(n_matrices):
        tangent_vectors_cmsis[i] = tangent_space_cmsis(P_cmsis, spd_matrices[i, :, :])

    return tangent_vectors_cmsis


def tangent_space_relative_to_mean_cmsis(spd_matrices):
    # Compute Riemannian mean using CMSIS-DSP implementation
    P_cmsis, _ = riemann_mean_cmsis(spd_matrices)

    # Compute tangent vectors using all three methods
    n_matrices = np.shape(spd_matrices)[0]

    # 6 unique elements for 3x3 matrix
    tangent_vectors_cmsis = np.zeros((n_matrices, 6))
    for i in range(n_matrices):
        tangent_vectors_cmsis[i] = tangent_space_cmsis(P_cmsis, spd_matrices[i, :, :])

    return tangent_vectors_cmsis

def tangent_space_cmsis(P, P_i):
    """
    Compute the tangent space mapping using the Riemannian metric with CMSIS-DSP.
    Uses CMSIS-DSP for matrix operations while keeping initialization in NumPy.
    Assumes py_svd3 returns sorted eigenvalues.

    Parameters:
    P : ndarray, shape (n, n)
        Reference point (Riemannian mean)
    P_i : ndarray, shape (n, n)
        SPD matrix to map to tangent space

    Returns:
    s_i : ndarray
        Tangent vector (vectorized upper triangular elements)
    """
    # Constants
    n = 3
    idx = np.triu_indices_from(np.empty((n, n)))
    coeffs = (np.sqrt(2) * np.triu(np.ones((n, n), dtype=np.float32), 1) + np.eye(n, dtype=np.float32))[idx]

    # Step 1: Compute P^(-1/2) using C-based SVD
    U, S, V = svd3(P)
    eigvals = np.diag(S)
    signs = np.diag(np.sign(U.T @ V))
    eigvals = eigvals * signs

    # Compute P^(-1/2) using CMSIS-DSP
    isqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
    P_isqrt = dsp.arm_mat_mult_f32(U, isqrt_diag)[1]
    P_isqrt = dsp.arm_mat_mult_f32(P_isqrt, U.T)[1]

    # Step 2: Compute whitened matrix using CMSIS-DSP
    temp = dsp.arm_mat_mult_f32(P_isqrt, P_i)[1]
    whitened = dsp.arm_mat_mult_f32(temp, P_isqrt)[1]

    # Ensure symmetry using CMSIS-DSP
    whitened_transposed = dsp.arm_mat_trans_f32(whitened)[1]
    whitened = dsp.arm_mat_add_f32(whitened, whitened_transposed)[1]
    whitened = dsp.arm_mat_scale_f32(whitened, 0.5)[1]

    # Step 3: Compute matrix logarithm using C-based SVD
    Uw, Sw, Vw = svd3(whitened)
    signs_w = np.diag(np.sign(Uw.T @ Vw))
    eigvals_w = np.diag(Sw) * signs_w

    # Compute matrix logarithm using CMSIS-DSP
    log_diag = np.diag(dsp.arm_vlog_f32(eigvals_w))  # diag on vector creates a zeros mat with diag values
    temp = dsp.arm_mat_mult_f32(Uw, log_diag)[1]
    log_whitened = dsp.arm_mat_mult_f32(temp, Uw.T)[1]

    # Extract upper triangular elements (back to NumPy for this part)
    s_i = dsp.arm_mult_f32(coeffs, log_whitened[idx[0], idx[1]])

    return s_i