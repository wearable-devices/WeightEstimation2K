import tensorflow as tf

# def get_spd_matrices_fixed_point(data):
#     # batch_size, channels, window_size = data.shape
#
#     batch_size = tf.shape(data)[0]  # Get dynamic batch size
#     channels = tf.shape(data)[1]  # Get dynamic channels
#     window_size = tf.shape(data)[2]
#     output_matrices = tf.zeros((batch_size, channels, channels), dtype=tf.float32)
#
#     for batch in range(batch_size):
#         batch_data = data[batch]
#         centered_data = center_data(batch_data)
#         sample_cov = compute_sample_covariance(centered_data, window_size)
#         target = compute_shrinkage_target(sample_cov, channels)
#         alpha = estimate_shrinkage_intensity(centered_data, sample_cov, target, window_size)
#         shrunk_cov = apply_shrinkage(sample_cov, target, alpha)
#         output_matrices[batch] = shrunk_cov
#
#     return output_matrices


def get_spd_matrices_fixed_point(data):
    # batch_size = tf.shape(data)[0]  # Get dynamic batch size
    channels = tf.shape(data)[1]  # Get dynamic channels
    window_size = tf.shape(data)[2]  # Get dynamic window size

    # Use tf.map_fn to process each batch item
    def process_single_batch(batch_data):
        centered_data = center_data(batch_data)
        sample_cov = compute_sample_covariance(centered_data, window_size)
        target = compute_shrinkage_target(sample_cov, channels)
        alpha = estimate_shrinkage_intensity(centered_data, sample_cov, target, window_size)
        shrunk_cov = apply_shrinkage(sample_cov, target, alpha)
        return shrunk_cov

    output_matrices = tf.map_fn(process_single_batch, data, dtype=data.dtype)
    return output_matrices

def center_data(batch_data):
    means = tf.reduce_mean(batch_data, axis=1, keepdims=True)
    return batch_data - means

def compute_sample_covariance(centered_data, window_size):
    return tf.matmul(centered_data, centered_data, transpose_b=True) / tf.cast(window_size, centered_data.dtype)#np.dot(centered_data, centered_data.T) / window_size

def compute_shrinkage_target(sample_cov, channels):
    trace_cov = tf.linalg.trace(sample_cov)#np.trace(sample_cov)
    # return (trace_cov / channels) * tf.eye(channels)#np.eye(channels)
    identity_matrix = tf.eye(tf.shape(sample_cov)[0], dtype=trace_cov.dtype)
    return (trace_cov / tf.cast(channels, trace_cov.dtype)) * identity_matrix

# def estimate_shrinkage_intensity(centered_data, sample_cov, target, window_size):
#     F_norm2 = tf.reduce_sum((sample_cov - target) ** 2)#np.sum((sample_cov - target) ** 2)
#     var_sum = 0
#
#     for i in range(window_size):
#         observation = centered_data[:, i:i+1]
#         outer_prod = tf.matmul(observation, observation, transpose_b=True)#np.dot(observation, observation.T)
#         diff = outer_prod - sample_cov
#         var_sum += tf.reduce_sum(diff ** 2)#np.sum(diff ** 2)
#
#     var_est = var_sum / (window_size ** 2)
#     return min(1, var_est / F_norm2) if F_norm2 > 0 else 0


def estimate_shrinkage_intensity(centered_data, sample_cov, target, window_size):
    F_norm2 = tf.reduce_sum((sample_cov - target) ** 2)

    # Vectorized computation instead of Python for loop
    # centered_data shape: [channels, window_size]
    # We want to compute outer product for each time point

    # Transpose to get [window_size, channels]
    data_transposed = tf.transpose(centered_data)  # Shape: [window_size, channels]

    # Expand dims to compute outer products: [window_size, channels, 1] and [window_size, 1, channels]
    data_expanded_1 = tf.expand_dims(data_transposed, axis=2)  # Shape: [window_size, channels, 1]
    data_expanded_2 = tf.expand_dims(data_transposed, axis=1)  # Shape: [window_size, 1, channels]

    # Compute outer products for all time points at once
    outer_prods = tf.matmul(data_expanded_1, data_expanded_2)  # Shape: [window_size, channels, channels]

    # Compute differences from sample covariance
    sample_cov_expanded = tf.expand_dims(sample_cov, 0)  # Shape: [1, channels, channels]
    diffs = outer_prods - sample_cov_expanded  # Broadcasting: [window_size, channels, channels]

    # Sum over all differences
    var_sum = tf.reduce_sum(diffs ** 2)
    var_est = var_sum / tf.cast(window_size ** 2, var_sum.dtype)

    # Return shrinkage intensity
    alpha = tf.cond(
        F_norm2 > 0,
        lambda: tf.minimum(1.0, var_est / F_norm2),
        lambda: 0.0
    )
    return alpha

def apply_shrinkage(sample_cov, target, alpha):
    return (1 - alpha) * sample_cov + alpha * target

# For testing
import pandas as pd

def get_framed_snc_data_from_file(file_path, window_size, frame_step, apply_zpk2sos_filter=False):
    data = pd.read_csv(file_path)

    # Extract sensors
    snc1 = data['Snc1'].values
    snc2 = data['Snc2'].values
    snc3 = data['Snc3'].values


    # if apply_zpk2sos_filter:
    #     snc1 = zpk2sos_filter(snc1)
    #     snc2 = zpk2sos_filter(snc2)
    #     snc3 = zpk2sos_filter(snc3)

    conc_snc = tf.concat([tf.expand_dims(snc1, axis=0),
                        tf.expand_dims(snc2, axis=0),
                        tf.expand_dims(snc3, axis=0)], axis=0)

    framed_data = tf.signal.frame(
                            conc_snc,
                            window_size,
                            frame_step,
                            pad_end=False,
                            pad_value=0,
                            axis=-1,
                            name=None
                        )

    framed_data = tf.transpose(framed_data, perm=(1,0,2))

    return framed_data, snc1, snc2, snc3#, snc_button
if __name__ == "__main__":
    window_size = 500
    frame_step = 50

    # Load data
    file = '/media/wld-algo-6/Storage/SortedCleaned/Alisa/press_release/Alisa_3_press_0_Nominal_TableTop_M.csv'
    test_files = [file]

    # Get labeled data from csv files
    framed_data, snc1, snc2, snc3 = get_framed_snc_data_from_file(file,window_size=window_size,
                                                            frame_step=frame_step,
                                                                apply_zpk2sos_filter=False)

    spd_matrices = get_spd_matrices_fixed_point(framed_data)

    ttt = 1
