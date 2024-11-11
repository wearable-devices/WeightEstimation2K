import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from db_generators.generators import MatchingLearningGenerator
# from db_generators.get_db import get_weight_file_from_dir
from scipy import special, stats

def logit(p):
    return tf.math.log(p / (1 - p))


def gaussian(x, mu, sigma):
    # Convert all inputs to float32
    x = tf.cast(x, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)

    return tf.exp(-tf.square(x - mu) / (2 * tf.square(sigma))) / (sigma * tf.sqrt(2 * tf.cast(np.pi, tf.float32)))


def find_interval(mu, sigma, p=0.5):
    # Find k for standard normal
    k = np.sqrt(2) * special.erfinv(p)

    # Calculate interval
    a = mu - k * sigma
    b = mu + k * sigma

    return a, b, k*sigma

def find_max_sigma(p=0.5, max_weight = 2.5):
    k = np.sqrt(2) * special.erfinv(p)
    return  max_weight/k


def batch_gaussian_distribution(input_tensor, smpl_rate, max_weight=1.0, normalize=True):
    """
    Generates Gaussian distributions for a batch of mu and sigma values.

    Args:
        input_tensor: tf.Tensor of shape (batch_size, 2) where [:, 0] is mu and [:, 1] is sigma
        smpl_rate: int, number of points to sample for each Gaussian
        max_weight: float, maximum x value for the linspace

    Returns:
        tf.Tensor of shape (batch_size, smpl_rate) containing Gaussian distributions
    """
    # Ensure input tensor is float32
    input_tensor = tf.cast(input_tensor, tf.float32)

    # Extract mu and sigma from input tensor
    mu = input_tensor[:, 0]  # Shape: (batch_size,)
    sigma = input_tensor[:, 1]  # Shape: (batch_size,)

    # Create x values
    x_values = tf.linspace(0.0, max_weight, smpl_rate)  # Shape: (smpl_rate,)

    # Reshape tensors for broadcasting
    # mu: (batch_size, 1)
    # sigma: (batch_size, 1)
    # x_values: (1, smpl_rate)
    mu = tf.expand_dims(mu, axis=1)
    sigma = tf.expand_dims(sigma, axis=1)
    x_values = tf.expand_dims(x_values, axis=0)

    # Calculate Gaussian distribution for all x values and all mu/sigma pairs
    result = gaussian(x_values, mu, sigma)  # Shape: (batch_size, smpl_rate)

    # Apply softmax normalization if requested
    if normalize:
        result = tf.nn.softmax(result, axis=-1)

    return result


def transform_tensor_with_gaussian(tensor, smpl_rate, max_weight, sigma=0.5):
    """
    Transform a tensor of weights into a Gaussian distribution representation.

    This function takes a tensor of weights and transforms each weight into a Gaussian
    distribution. The resulting tensor represents each weight as a series of points
    sampled from its corresponding Gaussian distribution.

    Args:
        tensor (tf.Tensor): Input tensor of weights with shape [batch_size, label_num, 1].
        smpl_rate (int): Number of samples to generate for each weight's Gaussian distribution.
        max_weight (float): Maximum weight value, used to determine the range of x values.
        sigma (float, optional): Standard deviation of the Gaussian distribution. Defaults to 0.5.

    Returns:
        tf.Tensor: Transformed tensor with shape [batch_size, label_num, smpl_rate], where each
                   weight is represented by `smpl_rate` points sampled from its Gaussian distribution.

    Note:
        - The function uses TensorFlow operations for efficient computation.
        - Input tensor is cast to float32 for consistency.
        - The Gaussian distribution is centered around each weight value (mu).
        - The x-axis range for sampling is from 0 to max_weight.
    """

    # Ensure tensor is float32
    tensor = tf.cast(tensor, tf.float32)

    # Get the shape of the input tensor
    shape = tf.shape(tensor)
    batch_size, label_num = shape[0], shape[1]

    # Create x values for sampling
    x_values = tf.linspace(0.0, max_weight, smpl_rate)

    # Reshape tensor to [batch_size * label_num, 1]
    flattened_tensor = tf.reshape(tensor, [-1, 1])

    # Create a meshgrid of mu values and x values
    mu_grid = tf.tile(flattened_tensor, [1, smpl_rate])
    x_grid = tf.tile(tf.expand_dims(x_values, 0), [tf.shape(flattened_tensor)[0], 1])

    # Apply Gaussian function
    y_values = gaussian(x_grid, mu=mu_grid, sigma=sigma)

    # Reshape back to [batch_size, label_num, smpl_rate]
    new_tensor = tf.reshape(y_values, [batch_size, label_num, smpl_rate])

    return new_tensor


def create_vector_tensor(value_list, batch_size, max_value=8):
    """
    Create a tensor of 2D vectors based on a list of values.

    This function takes a list of values and returns a tensor of shape
    (batch_size, len(value_list), 2). Each 2D vector in the output corresponds
    to a point on a quarter circle arc from (1,0) to (0,1), with the position
    determined by the corresponding value in the input list.

    Args:
    value_list (list): List of integer values. Should include 0 and the maximum value.
    batch_size (int): The desired batch size for the output tensor.

    Returns:
    tf.Tensor: Output tensor of shape (batch_size, len(value_list), 2).
               For each value in the input list, contains a 2D vector (x, y) where:
               - The minimum value (0) corresponds to (1, 0)
               - The maximum value corresponds to (0, 1)
               - Intermediate values are positioned on the arc between these points

    Note:
    - The function assumes the minimum value in the list is 0 and the maximum value is 8.
    - If these assumptions don't hold, adjust the angle calculation accordingly.
    """
    # max_value = 8  # Assuming the maximum value is 8
    num_values = len(value_list)

    # Convert value_list to a tensor
    value_tensor = tf.constant(value_list, dtype=tf.float32)

    # Calculate angles for all values at once
    # Normalize values to range [0, 1] and then multiply by pi/2
    angles = (value_tensor / max_value) * (np.pi / 2)

    # Calculate cos and sin for all angles at once
    cos_values = tf.cos(angles)
    sin_values = tf.sin(angles)

    # Stack cos and sin values to create 2D vectors
    vectors = tf.stack([cos_values, sin_values], axis=-1)

    # Normalize the vectors
    vector_magnitudes = tf.norm(vectors, axis=-1, keepdims=True)
    normalized_vectors = vectors / vector_magnitudes

    # Broadcast to batch size
    output_tensor = tf.broadcast_to(normalized_vectors, (batch_size, num_values, 2))

    return output_tensor


def transform_tensor(a, max_weight):
    """
        Transform a 3D tensor into a new tensor with 2D vectors representing points on a quarter circle.

        This function takes a tensor of shape (batch_size, label_num, 1) and returns a new tensor
        of shape (batch_size, label_num, 2). Each 2D vector in the output corresponds to a point
        on a quarter circle arc from (1,0) to (0,1), evenly distributed based on the label number.

        Args:
        a (tf.Tensor): Input tensor of shape (batch_size, label_num, 1).
                       The actual values in this tensor are not used, only its shape.

        Returns:
        tf.Tensor: Output tensor of shape (batch_size, label_num, 2).
                   For each label, contains a 2D vector (x, y) where:
                   - Label 0 corresponds to (1, 0)
                   - The last label corresponds to (0, 1)
                   - Intermediate labels are evenly spaced on the arc between these points

        Note:
        - The function uses the dtype of the input tensor for all calculations.
        - np.pi is used to calculate the angle step, ensure numpy is imported.
        """
    batch_size, label_num = a.shape

    # Create a tensor of shape (batch_size, label_num, 2)
    # b = tf.zeros((batch_size, label_num, 2), dtype=a.dtype)

    # Calculate the angle step
    angle_step = tf.constant(np.pi / 2 / (max_weight - 1), dtype=a.dtype)

    # Create a range tensor for label indices
    label_indices = tf.range(label_num, dtype=a.dtype)

    # Calculate angles for all labels at once
    angles = label_indices * angle_step

    # Calculate cos and sin for all angles at once
    cos_values = tf.cos(angles)
    sin_values = tf.sin(angles)

    # Stack cos and sin values
    vectors = tf.stack([cos_values, sin_values], axis=-1)

    # Broadcast to batch size
    b = tf.broadcast_to(vectors, (batch_size, label_num, 2))

    return b


def weight_iterpretation(weight_tensor, sampling_rate=6):
    '''
    takes weight_tensor of shape (batch_size,dim,1) and returns tensor of shape  (batch_size,dim,sampling_rate)
    map each weight to the vector of distribution and returns  a tensor'''
    s = 1


if __name__ == "__main__":
    window_size = 306
    samples_per_weight = 2
    smpl_rate = 100
    sigma = 1.5
    train_dir = '/home/wld-algo-5/Data/17_7_2024 data base/Train012468'
    # train_persons_dict = get_weight_file_from_dir(train_dir)

    # train_ds = MatchingLearningGenerator(persons_dict_for_labeled_data=train_persons_dict,
    #                                      persons_dict_for_query_data=train_persons_dict,
    #                                      window_size=window_size, k_shot=10, q_query=1,
    #                                      samples_per_weight=samples_per_weight, persons_per_batch=1,
    #                                      tasks_per_person=4)

    # batch = train_ds.__getitem__(0)
    # support_labels = batch[0][6]

    # prob_support_labels = transform_tensor_with_gaussian(support_labels, smpl_rate, 8, sigma=sigma)#[0, :, :]
    c = 8 + 0.1
    w = 1
    # Create a range of x values
    x = tf.linspace(0, 8, 100)

    # Plot the results
    # plt.figure(figsize=(10, 6))
    # for w in range(int(c)+1):
    #     y = gaussian(logit(x / c), mu=logit(w / c), sigma=sigma)
    #     plt.plot(x, y)
    #
    # plt.legend()
    # plt.title('Normal Distribution (μ=0, σ=1)')
    # plt.xlabel('x')
    # plt.ylabel('Probability Density')
    # plt.grid(True)
    # plt.show()

    weights = tf.constant([[[0], [0.5], [1], [2.5]]])  # shape: (1, 4, 1)
    max_weight = 2.5
    # new_tensor = transform_tensor_with_gaussian(weights, smpl_rate, max_weight, sigma=0.5)
    x_values = tf.linspace(0.0, max_weight, smpl_rate)
    # # Plot the results
    # plt.figure(figsize=(10, 6))
    # for w in range(4):
    #     y = new_tensor[0,w,:]
    #     plt.plot(x_values, y)
    #
    # plt.legend()
    # plt.title('Normal Distribution (μ=0, σ=1)')
    # plt.xlabel('x')
    # plt.ylabel('Probability Density')
    # plt.grid(True)
    # plt.show()

    # Create sample input
    batch_size = 3
    input_data = tf.constant([
        # [0.5, 0.1],  # mu=0.5, sigma=0.1
        # [0.3, 0.2],  # mu=0.3, sigma=0.2
        # [0.7, 0.15],  # mu=0.7, sigma=0.15
        [2.5, 0.3],
        [0, 0.001],
        [1, 3]
    ])

    # Generate distributions
    distributions_normalized = batch_gaussian_distribution(input_data, smpl_rate=100, max_weight=max_weight, normalize=True)
    distributions_not_normalized = batch_gaussian_distribution(input_data, smpl_rate=100, max_weight=max_weight,
                                                           normalize=False)
    # Create figure and axes with 2 rows, 3 columns
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    for w in range(input_data.shape[0]):
        y_norm = distributions_normalized[w, :]
        axes[0].plot(x_values, y_norm)

        y_not_norm = distributions_not_normalized[w, :]
        axes[1].plot(x_values, y_not_norm)

    plt.legend()
    plt.title('Normal Distribution (μ=0, σ=1)')
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.show()






