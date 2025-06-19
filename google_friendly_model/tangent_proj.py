
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable

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

# def tangent_space_cmsis(P, P_i):
#     """
#     Compute the tangent space mapping using the Riemannian metric with CMSIS-DSP.
#     Uses CMSIS-DSP for matrix operations while keeping initialization in NumPy.
#     Assumes py_svd3 returns sorted eigenvalues.
#
#     Parameters:
#     P : ndarray, shape (n, n)
#         Reference point (Riemannian mean)
#     P_i : ndarray, shape (n, n)
#         SPD matrix to map to tangent space
#
#     Returns:
#     s_i : ndarray
#         Tangent vector (vectorized upper triangular elements)
#     """
#     # Constants
#     n = 3
#     idx = np.triu_indices_from(np.empty((n, n)))
#     coeffs = (np.sqrt(2) * np.triu(np.ones((n, n), dtype=np.float32), 1) + np.eye(n, dtype=np.float32))[idx]
#
#     # Step 1: Compute P^(-1/2) using C-based SVD
#     # U, S, V = svd3(P)
#     S, U, V = tf.linalg.svd(P, full_matrices=True, compute_uv=True)
#     eigvals = np.diag(S)
#     signs = np.diag(np.sign(U.T @ V))
#     eigvals = eigvals * signs
#
#     # Compute P^(-1/2) using CMSIS-DSP
#     isqrt_diag = np.diag(1.0 / np.sqrt(eigvals))
#     P_isqrt = dsp.arm_mat_mult_f32(U, isqrt_diag)[1]
#     P_isqrt = dsp.arm_mat_mult_f32(P_isqrt, U.T)[1]
#
#     # Step 2: Compute whitened matrix using CMSIS-DSP
#     temp = dsp.arm_mat_mult_f32(P_isqrt, P_i)[1]
#     whitened = dsp.arm_mat_mult_f32(temp, P_isqrt)[1]
#
#     # Ensure symmetry using CMSIS-DSP
#     whitened_transposed = dsp.arm_mat_trans_f32(whitened)[1]
#     whitened = dsp.arm_mat_add_f32(whitened, whitened_transposed)[1]
#     whitened = dsp.arm_mat_scale_f32(whitened, 0.5)[1]
#
#     # Step 3: Compute matrix logarithm using C-based SVD
#     Uw, Sw, Vw = svd3(whitened)
#     signs_w = np.diag(np.sign(Uw.T @ Vw))
#     eigvals_w = np.diag(Sw) * signs_w
#
#     # Compute matrix logarithm using CMSIS-DSP
#     log_diag = np.diag(dsp.arm_vlog_f32(eigvals_w))  # diag on vector creates a zeros mat with diag values
#     temp = dsp.arm_mat_mult_f32(Uw, log_diag)[1]
#     log_whitened = dsp.arm_mat_mult_f32(temp, Uw.T)[1]
#
#     # Extract upper triangular elements (back to NumPy for this part)
#     s_i = dsp.arm_mult_f32(coeffs, log_whitened[idx[0], idx[1]])
#
#     return s_i

# class TangentSpaceLayer(tf.keras.layers.Layer):
#     """
#     TensorFlow layer for computing tangent space mapping of SPD matrices.
#
#     This layer computes the tangent space projection using the Riemannian metric,
#     converting SPD matrices to tangent vectors by mapping them to the tangent space
#     at a reference point (Riemannian mean).
#
#     Input shape: (batch_size, n_channels, n_channels) - batched SPD matrices
#     Output shape: (batch_size, n_features) - where n_features = n_channels * (n_channels + 1) / 2
#     """
#
#     def __init__(self, n_channels=3, epsilon=1e-8, **kwargs):
#         super(TangentSpaceLayer, self).__init__(**kwargs)
#         self.n_channels = n_channels
#         self.epsilon = epsilon
#
#         # Precompute indices and coefficients for upper triangular extraction
#         self.upper_tri_indices = np.triu_indices(n_channels)
#         coeffs_matrix = (np.sqrt(2) * np.triu(np.ones((n_channels, n_channels), dtype=np.float32), 1) +
#                          np.eye(n_channels, dtype=np.float32))
#         self.coeffs = tf.constant(coeffs_matrix[self.upper_tri_indices], dtype=tf.float32)
#
#         # Initialize reference point (will be updated during training or set externally)
#         self.reference_point = self.add_weight(
#             name='reference_point',
#             shape=(n_channels, n_channels),
#             initializer='identity',
#             trainable=False
#         )
#
#     def set_reference_point(self, P):
#         """Set the reference point for tangent space mapping"""
#         self.reference_point.assign(P)
#
#     def build(self, input_shape):
#         super(TangentSpaceLayer, self).build(input_shape)
#
#     def call(self, inputs, training=None):
#         """
#         Apply tangent space mapping to batched SPD matrices.
#
#         Args:
#             inputs: Tensor of shape (batch_size, n_channels, n_channels)
#
#         Returns:
#             Tensor of shape (batch_size, n_features) containing tangent vectors
#         """
#         batch_size = tf.shape(inputs)[0]
#
#         # Compute P^(-1/2) where P is the reference point
#         P_isqrt = self._compute_matrix_power(self.reference_point, -0.5)
#
#         # Process each matrix in the batch
#         tangent_vectors = tf.map_fn(
#             lambda P_i: self._single_tangent_mapping(P_i, P_isqrt),
#             inputs,
#             fn_output_signature=tf.TensorSpec(shape=(len(self.upper_tri_indices[0]),), dtype=tf.float32),
#             parallel_iterations=32
#         )
#
#         return tangent_vectors
#
#     def _single_tangent_mapping(self, P_i, P_isqrt):
#         """Apply tangent space mapping to a single SPD matrix"""
#         # Step 1: Compute whitened matrix: P^(-1/2) * P_i * P^(-1/2)
#         temp = tf.linalg.matmul(P_isqrt, P_i)
#         whitened = tf.linalg.matmul(temp, P_isqrt)
#
#         # Ensure symmetry
#         whitened = 0.5 * (whitened + tf.linalg.matrix_transpose(whitened))
#
#         # Step 2: Compute matrix logarithm
#         log_whitened = self._matrix_logarithm(whitened)
#
#         # Step 3: Extract upper triangular elements with coefficients
#         upper_tri_elements = tf.gather_nd(log_whitened,
#                                           tf.stack([self.upper_tri_indices[0],
#                                                     self.upper_tri_indices[1]], axis=1))
#
#         # Apply coefficients
#         s_i = self.coeffs * upper_tri_elements
#
#         return s_i
#
#     def _compute_matrix_power(self, matrix, power):
#         """Compute matrix^power using eigendecomposition"""
#         # Add small epsilon to diagonal for numerical stability
#         matrix_stable = matrix + self.epsilon * tf.eye(self.n_channels, dtype=matrix.dtype)
#
#         # Compute eigendecomposition
#         eigenvalues, eigenvectors = tf.linalg.eigh(matrix_stable)
#
#         # Ensure positive eigenvalues
#         eigenvalues = tf.maximum(eigenvalues, self.epsilon)
#
#         # Compute eigenvalues^power
#         powered_eigenvalues = tf.pow(eigenvalues, power)
#
#         # Reconstruct matrix: V * D^power * V^T
#         result = tf.linalg.matmul(
#             eigenvectors,
#             tf.linalg.matmul(
#                 tf.linalg.diag(powered_eigenvalues),
#                 tf.linalg.matrix_transpose(eigenvectors)
#             )
#         )
#
#         return result
#
#     def _matrix_logarithm(self, matrix):
#         """Compute matrix logarithm using eigendecomposition"""
#         # Add small epsilon to diagonal for numerical stability
#         matrix_stable = matrix + self.epsilon * tf.eye(self.n_channels, dtype=matrix.dtype)
#
#         # Compute eigendecomposition
#         eigenvalues, eigenvectors = tf.linalg.eigh(matrix_stable)
#
#         # Ensure positive eigenvalues
#         eigenvalues = tf.maximum(eigenvalues, self.epsilon)
#
#         # Compute log of eigenvalues
#         log_eigenvalues = tf.math.log(eigenvalues)
#
#         # Reconstruct matrix: V * log(D) * V^T
#         result = tf.linalg.matmul(
#             eigenvectors,
#             tf.linalg.matmul(
#                 tf.linalg.diag(log_eigenvalues),
#                 tf.linalg.matrix_transpose(eigenvectors)
#             )
#         )
#
#         return result
#
#     def get_config(self):
#         config = super(TangentSpaceLayer, self).get_config()
#         config.update({
#             'n_channels': self.n_channels,
#             'epsilon': self.epsilon
#         })
#         return config

# @register_keras_serializable(package='weight_estimation')
# class TangentSpaceLayer(tf.keras.layers.Layer): # is working but affects the speed of learning process in a bad way
#     """
#     TensorFlow Lite compatible TangentSpaceLayer - Fixed for all TF versions
#
#     This version avoids tf.eye with batch_size parameter and other version-specific issues
#     """
#
#     def __init__(self, epsilon=1e-6, **kwargs):
#         """
#         Initialize the layer
#
#         Args:
#             epsilon: Small regularization value for numerical stability
#         """
#         super().__init__(**kwargs)
#         self.epsilon = epsilon
#
#     def build(self, input_shape):
#         """Build the layer and validate input shape"""
#         super().build(input_shape)
#
#         # Validate input
#         if len(input_shape) != 3:
#             raise ValueError(f"Expected 3D input (batch, n, n), got: {input_shape}")
#
#         if input_shape[-1] != input_shape[-2]:
#             raise ValueError(f"Expected square matrices, got: {input_shape}")
#
#         self.n = input_shape[-1]
#
#         # Calculate output dimension (upper triangular elements)
#         self.output_dim = self.n * (self.n + 1) // 2
#
#         # Pre-create identity matrix (avoids batch_size parameter issue)
#         self.identity_matrix = tf.Variable(
#             initial_value=tf.eye(self.n, dtype=tf.float32),
#             trainable=False,
#             name='identity_matrix'
#         )
#
#     def _create_batch_identity(self, batch_size, dtype):
#         """
#         Create batched identity matrices without using batch_size parameter
#         """
#         # Method 1: Use tile to create batch of identity matrices
#         identity = tf.cast(self.identity_matrix, dtype)
#         identity = tf.expand_dims(identity, 0)  # Add batch dimension
#         batched_identity = tf.tile(identity, [batch_size, 1, 1])
#         return batched_identity
#
#     def _make_spd(self, matrices):
#         """
#         Ensure matrices are symmetric positive definite
#         """
#         batch_size = tf.shape(matrices)[0]
#
#         # Make symmetric
#         symmetric = 0.5 * (matrices + tf.transpose(matrices, perm=[0, 2, 1]))
#
#         # Create batched identity matrix
#         batched_identity = self._create_batch_identity(batch_size, matrices.dtype)
#
#         # Add regularization to diagonal
#         spd_matrices = symmetric + self.epsilon * batched_identity
#
#         return spd_matrices
#
#     def _matrix_log_approximation(self, spd_matrices):
#         """
#         Approximate matrix logarithm using Cholesky decomposition
#         """
#         try:
#             # Cholesky decomposition: A = L * L^T
#             L = tf.linalg.cholesky(spd_matrices)
#
#             # Extract diagonal elements
#             diag_elements = tf.linalg.diag_part(L)
#
#             # Compute log of diagonal elements (main contribution to matrix log)
#             log_diag = tf.math.log(tf.maximum(diag_elements, self.epsilon))
#
#             # Create approximate matrix logarithm
#             # Method: Use the structure of L but with log-transformed diagonal
#             log_L = tf.linalg.set_diag(L, log_diag)
#
#             # Approximate log(A) ≈ log_L + log_L^T (simplified)
#             log_matrices = log_L + tf.transpose(log_L, perm=[0, 2, 1])
#
#             # Apply scaling factor to improve approximation
#             log_matrices = 0.5 * log_matrices
#
#             return log_matrices
#
#         except Exception as e:
#             # Fallback: simple approximation if Cholesky fails
#             return self._matrix_log_fallback(spd_matrices)
#
#     def _matrix_log_fallback(self, spd_matrices):
#         """
#         Fallback method for matrix logarithm approximation
#         """
#         batch_size = tf.shape(spd_matrices)[0]
#
#         # Create identity matrix
#         batched_identity = self._create_batch_identity(batch_size, spd_matrices.dtype)
#
#         # Simple approximation: log(A) ≈ 2 * (A - I) / (A + I)
#         A_minus_I = spd_matrices - batched_identity
#         A_plus_I = spd_matrices + batched_identity
#
#         # Use solve instead of inverse for numerical stability
#         try:
#             log_approx = 2.0 * tf.linalg.solve(A_plus_I, A_minus_I)
#         except:
#             # Final fallback - just use A - I
#             log_approx = A_minus_I
#
#         return log_approx
#
#     def _vectorize_upper_triangular(self, matrices):
#         """
#         Extract upper triangular elements from symmetric matrices
#         """
#         # Handle different matrix sizes explicitly for TFLite compatibility
#         if self.n == 2:
#             elements = [
#                 matrices[:, 0, 0],  # (0,0)
#                 matrices[:, 0, 1],  # (0,1)
#                 matrices[:, 1, 1],  # (1,1)
#             ]
#         elif self.n == 3:
#             elements = [
#                 matrices[:, 0, 0],  # (0,0)
#                 matrices[:, 0, 1],  # (0,1)
#                 matrices[:, 0, 2],  # (0,2)
#                 matrices[:, 1, 1],  # (1,1)
#                 matrices[:, 1, 2],  # (1,2)
#                 matrices[:, 2, 2],  # (2,2)
#             ]
#         elif self.n == 4:
#             elements = [
#                 matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2], matrices[:, 0, 3],
#                 matrices[:, 1, 1], matrices[:, 1, 2], matrices[:, 1, 3],
#                 matrices[:, 2, 2], matrices[:, 2, 3],
#                 matrices[:, 3, 3],
#             ]
#         else:
#             # General case for larger matrices
#             elements = []
#             for i in range(self.n):
#                 for j in range(i, self.n):
#                     elements.append(matrices[:, i, j])
#
#         return tf.stack(elements, axis=1)
#
#     def call(self, inputs):
#         """
#         Convert SPD matrices to tangent space representation
#
#         Args:
#             inputs: Batch of SPD matrices, shape (batch_size, n, n)
#
#         Returns:
#             Tangent space vectors, shape (batch_size, n*(n+1)/2)
#         """
#         # Ensure float32 for TFLite compatibility
#         matrices = tf.cast(inputs, tf.float32)
#
#         # Ensure matrices are SPD
#         spd_matrices = self._make_spd(matrices)
#
#         # Compute approximate matrix logarithm
#         log_matrices = self._matrix_log_approximation(spd_matrices)
#
#         # Vectorize to tangent space
#         tangent_vectors = self._vectorize_upper_triangular(log_matrices)
#
#         return tangent_vectors
#
#     def compute_output_shape(self, input_shape):
#         """Compute output shape"""
#         return (input_shape[0], self.output_dim)
#
#     def get_config(self):
#         """Get layer configuration for serialization"""
#         config = super().get_config()
#         config.update({'epsilon': self.epsilon})
#         return config
#
#     @classmethod
#     def from_config(cls, config):
#         """Create layer from configuration"""
#         return cls(**config)
#


@register_keras_serializable(package='weight_estimation')
class TangentSpaceLayer(tf.keras.layers.Layer):
    """
    Two-stage TangentSpaceLayer:
    1. Training: Uses true eigenvalue decomposition for optimal learning
    2. Deployment: Can be replaced with pre-computed approximation or converted differently
    """

    def __init__(self,
                 mode='training',  # 'training' or 'deployment'
                 epsilon=1e-8,
                 **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        self.epsilon = epsilon

    def build(self, input_shape):
        super().build(input_shape)

        if len(input_shape) != 3:
            raise ValueError(f"Expected 3D input, got: {input_shape}")

        self.n = int(input_shape[-1])
        self.output_dim = self.n * (self.n + 1) // 2

        print(f"TangentSpaceLayer mode: {self.mode}")
        print(f"Matrix size: {self.n}x{self.n}, Output dim: {self.output_dim}")

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        n = int(input_shape[-1])
        output_dim = n * (n + 1) // 2
        return (batch_size, output_dim)

    def _ensure_spd(self, matrices):
        """Minimal SPD regularization"""
        batch_size = tf.shape(matrices)[0]

        # Symmetrize
        symmetric = 0.5 * (matrices + tf.transpose(matrices, perm=[0, 2, 1]))

        # Add minimal regularization
        eye = tf.eye(self.n, dtype=matrices.dtype)
        eye_batch = tf.tile(tf.expand_dims(eye, 0), [batch_size, 1, 1])

        return symmetric + self.epsilon * eye_batch

    def _matrix_log_eigenvalue_exact(self, matrices):
        """
        EXACT matrix logarithm using eigenvalue decomposition
        This is what you want for training - preserves all geometric properties
        """
        # Eigenvalue decomposition for SPD matrices
        eigenvals, eigenvecs = tf.linalg.eigh(matrices) ### SELF ADJOINT

        # Ensure positive eigenvalues (numerical safety)
        eigenvals = tf.maximum(eigenvals, self.epsilon)

        # Compute logarithm of eigenvalues
        log_eigenvals = tf.math.log(eigenvals)

        # Reconstruct matrix logarithm: log(A) = V * diag(log(λ)) * V^T
        log_matrices = tf.linalg.matmul(
            eigenvecs,
            tf.linalg.matmul(
                tf.linalg.diag(log_eigenvals),
                tf.transpose(eigenvecs, perm=[0, 2, 1])
            )
        )

        return log_matrices

    def _vectorize_symmetric_matrix(self, matrices):
        """Extract upper triangular elements"""
        if self.n == 2:
            return tf.stack([
                matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 1, 1]
            ], axis=1)
        elif self.n == 3:
            return tf.stack([
                matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2],
                matrices[:, 1, 1], matrices[:, 1, 2], matrices[:, 2, 2]
            ], axis=1)
        elif self.n == 4:
            return tf.stack([
                matrices[:, 0, 0], matrices[:, 0, 1], matrices[:, 0, 2], matrices[:, 0, 3],
                matrices[:, 1, 1], matrices[:, 1, 2], matrices[:, 1, 3],
                matrices[:, 2, 2], matrices[:, 2, 3], matrices[:, 3, 3]
            ], axis=1)
        else:
            elements = []
            for i in range(self.n):
                for j in range(i, self.n):
                    elements.append(matrices[:, i, j])
            return tf.stack(elements, axis=1)

    def call(self, inputs, training=None):
        """
        Forward pass - uses EXACT eigenvalue method for training
        """
        matrices = tf.cast(inputs, tf.float32)
        spd_matrices = self._ensure_spd(matrices)

        # Always use exact eigenvalue method for training
        # This preserves the true Riemannian geometry
        log_matrices = self._matrix_log_eigenvalue_exact(spd_matrices)

        # Ensure symmetry (should already be, but numerical safety)
        log_matrices = 0.5 * (log_matrices + tf.transpose(log_matrices, perm=[0, 2, 1]))

        # Vectorize to tangent space
        tangent_vectors = self._vectorize_symmetric_matrix(log_matrices)
        tangent_vectors = tf.ensure_shape(tangent_vectors, (None, self.output_dim))

        return tangent_vectors

    def get_config(self):
        config = super().get_config()
        config.update({
            'mode': self.mode,
            'epsilon': self.epsilon
        })
        return config

# Utility function to compute Riemannian mean for setting reference point
def compute_riemannian_mean(spd_matrices, max_iterations=50, tolerance=1e-6):
    """
    Compute the Riemannian mean of a batch of SPD matrices.

    Args:
        spd_matrices: Tensor of shape (batch_size, n, n)
        max_iterations: Maximum number of iterations
        tolerance: Convergence tolerance

    Returns:
        Riemannian mean matrix of shape (n, n)
    """
    n_matrices, n, _ = spd_matrices.shape

    # Initialize with arithmetic mean
    mean_matrix = tf.reduce_mean(spd_matrices, axis=0)

    for i in range(max_iterations):
        # Compute mean in tangent space
        tangent_vectors = []
        mean_isqrt = tf.linalg.inv(tf.linalg.sqrtm(mean_matrix))

        for j in range(n_matrices):
            # Map to tangent space
            temp = tf.linalg.matmul(mean_isqrt, spd_matrices[j])
            whitened = tf.linalg.matmul(temp, mean_isqrt)
            log_matrix = tf.linalg.logm(whitened)
            tangent_vectors.append(log_matrix)

        # Average in tangent space
        tangent_mean = tf.reduce_mean(tf.stack(tangent_vectors), axis=0)

        # Map back to manifold
        mean_sqrt = tf.linalg.sqrtm(mean_matrix)
        exp_matrix = tf.linalg.expm(tangent_mean)
        new_mean = tf.linalg.matmul(mean_sqrt, tf.linalg.matmul(exp_matrix, mean_sqrt))

        # Check convergence
        if tf.linalg.norm(new_mean - mean_matrix) < tolerance:
            break

        mean_matrix = new_mean

    return mean_matrix


# test
from google_friendly_model.covariances import get_framed_snc_data_from_file, get_spd_matrices_fixed_point
if __name__ == "__main__":
    window_size = 500
    frame_step = 50

    # Load data
    file = '/media/wld-algo-6/Storage/SortedCleaned/Alisa/press_release/Alisa_3_press_0_Nominal_TableTop_M.csv'

    # Get labeled data from csv files
    framed_data, snc1, snc2, snc3 = get_framed_snc_data_from_file(file,window_size=window_size,
                                                            frame_step=frame_step,
                                                                apply_zpk2sos_filter=False)

    spd_matrices = get_spd_matrices_fixed_point(framed_data)

    # Apply tangent space mapping
    n_channels = 3
    tangent_layer = TangentSpaceLayer(n_channels=n_channels)
    tangent_features = tangent_layer(spd_matrices)

    ttt = 1