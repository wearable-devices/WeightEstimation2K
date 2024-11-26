import tensorflow as tf
import keras
import keras.ops as K
from custom.layers import DistanceLayer
# from keras.losses import Reduction

class WeightedMeanSquaredError(keras.losses.Loss):
    def __init__(self, weight_dict, #reduction=Reduction.AUTO,
                 name="weighted_mean_squared_error"):
        super().__init__( name=name)
        self.weight_dict = {float(k): tf.constant(v, dtype=tf.float32) for k, v in weight_dict.items()}

    def call(self, y_true, y_pred):
        # Calculate the squared difference
        y_true_cast = tf.cast(y_true, dtype=tf.float32)
        squared_diff = tf.square(y_pred - y_true_cast)

        # Calculate the weight using the dictionary
        weight = tf.zeros_like(y_true_cast)
        for k, v in self.weight_dict.items():
            weight += tf.where(tf.equal(y_true_cast, k), v, 0.0)

        # Apply the weight to the squared difference
        weighted_squared_diff = squared_diff * weight

        # Return the mean of the weighted squared differences
        return tf.reduce_mean(weighted_squared_diff)

    def get_config(self):
        config = super(WeightedMeanSquaredError, self).get_config()
        config.update({
            "weight_dict": {k: v.numpy() for k, v in self.weight_dict.items()}
        })
        return config


class GaussianNLLLoss(keras.losses.Loss):
    def __init__(self, eps=1e-6, name='gaussian_nll_loss'):
        super().__init__(name=name)
        self.eps = eps

    def call(self, y_true, y_pred):
        # Ensure y_true is the right shape
        y_true = K.cast(y_true, dtype='float32')

        # Split y_pred into mu and sigma
        # Assuming y_pred has shape (batch_size, 2)
        mu = y_pred[:, 0]  # First column is mu
        sigma = 1/y_pred[:, 1]  # Second column is sigma

        # Add small epsilon to avoid log(0) and division by 0
        sigma = sigma + self.eps

        # Compute negative log likelihood
        # Make sure all operations maintain the batch dimension
        nll = -K.log(sigma) - (K.square(y_true - mu)) / (2 * K.square(sigma))

        # Return mean over batch
        return K.mean(nll)

    def get_config(self):
        config = super().get_config()
        config.update({"eps": self.eps})
        return config

class GaussianCrossEntropyLoss(keras.losses.Loss):
    def  __init__(self, smpl_rate=100, max_weight=2.5, fixed_sigma=0.001,reduction=None,  normalize=False,
                 name='gaussian_cross_entropy'):
        """
        Initialize the Gaussian Cross Entropy Loss.

        Args:
            smpl_rate: int, number of points to sample for each Gaussian
            max_weight: float, maximum x value for the linspace
            fixed_sigma: float, sigma value to use for true values
            name: string, name of the loss function
        """
        super().__init__(name=name)
        self.smpl_rate = smpl_rate
        self.max_weight = max_weight
        self.fixed_sigma = fixed_sigma
        self.normalize = normalize

    def gaussian(self, x, mu, sigma):
        """Compute Gaussian distribution."""
        x = tf.cast(x, tf.float32)
        mu = tf.cast(mu, tf.float32)
        sigma = tf.cast(sigma, tf.float32)

        return tf.exp(-tf.square(x - mu) / (2 * tf.square(sigma))) / (
                    sigma * tf.sqrt(2 * tf.cast(3.14159, tf.float32)))

    def create_distribution(self, mu, sigma):
        """Create Gaussian distribution for given mu and sigma."""
        # Create x values
        x_values = tf.linspace(0.0, self.max_weight, self.smpl_rate)

        # Reshape tensors for broadcasting
        mu = tf.expand_dims(mu, axis=1)  # (batch_size, 1)
        sigma = tf.expand_dims(sigma, axis=1)  # (batch_size, 1)
        x_values = tf.expand_dims(x_values, axis=0)  # (1, smpl_rate)

        # Calculate distribution
        distribution = self.gaussian(x_values, mu, sigma)
        if self.normalize:
            distribution = tf.nn.softmax(distribution, axis=-1)

        # Apply softmax normalization
        return distribution#tf.nn.softmax(distribution, axis=-1)

    def call(self, y_true, y_pred):
        """
        Compute the loss between true and predicted distributions.

        Args:
            y_true: tensor of shape (batch_size,) containing true values
            y_pred: tensor of shape (batch_size, 2) containing predicted mu and sigma

        Returns:
            loss: scalar tensor
        """
        # Ensure inputs are float32
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Extract predicted mu and sigma
        # tf.print('y_pred', y_pred.shape)
        # tf.print('y_true', y_true.shape)
        pred_mu = y_pred[:, 0]
        pred_sigma = y_pred[:, 1]

        # tf.print('y_true.shape',y_true.shape)
        # tf.print('y_pred.shape', y_pred.shape)

        # Create true and predicted distributions
        true_distribution = self.create_distribution(y_true, tf.ones_like(y_true) * self.fixed_sigma)
        true_distribution = tf.nn.softmax(true_distribution, axis=-1)
        # true_distribution = tf.one_hot(tf.cast(y_true, tf.int32), self.smpl_rate)
        pred_distribution = self.create_distribution(pred_mu, pred_sigma)

        # Calculate cross entropy loss
        # Add small epsilon to avoid log(0)
        epsilon = 1e-7
        cross_entropy = -tf.reduce_sum(
            true_distribution * tf.math.log(pred_distribution + epsilon),
            axis=-1
        )

        cross_entropy = keras.losses.CategoricalCrossentropy()(true_distribution, pred_distribution)

        # Return mean loss across batch
        return tf.reduce_mean(cross_entropy)


def calculate_distances(y_pred, prototypes):
    """
    Calculate Euclidean distances between predictions and prototypes using Keras/TF.

    Args:
        y_pred: Tensor of shape (batch_size, embd_dim)
        prototypes: List of num_pers tensors, each of shape (embd_dim,)

    Returns:
        Tensor of shape (batch_size, num_pers) containing distances
    """
    # Convert list of prototypes to a tensor
    proto_tensor = tf.stack(prototypes)  # Shape: (num_pers, embd_dim)

    # Expand dimensions to enable broadcasting
    # y_pred: (batch_size, 1, embd_dim)
    # proto_tensor: (1, num_pers, embd_dim)
    y_pred_expanded = tf.expand_dims(y_pred, axis=1)
    proto_expanded = tf.expand_dims(proto_tensor, axis=0)

    # Calculate squared differences
    squared_diff = tf.square(y_pred_expanded - proto_expanded)

    # Sum along embedding dimension and take square root
    distances = tf.sqrt(tf.reduce_sum(squared_diff, axis=2))

    return distances


def analyze_prototype_distances(prototypes):
    """
    Calculate distances between all pairs of prototypes and find the minimum distance.
    Using Keras 3 operations.

    Args:
        prototypes: List of tensors, each of shape (embd_dim,)

    Returns:
        min_distance: Minimum distance between any pair of prototypes
        closest_pair: Tuple of indices (i, j) of the closest prototypes
        all_distances: Matrix of all pairwise distances
    """
    # Convert list of prototypes to a tensor
    proto_tensor = keras.ops.stack(prototypes)  # Shape: (num_protos, embd_dim)
    num_protos = len(prototypes)

    # Calculate pairwise distances between all prototypes
    # Expand dimensions for broadcasting
    p1 = keras.ops.expand_dims(proto_tensor, axis=0)  # Shape: (1, num_protos, embd_dim)
    p2 = keras.ops.expand_dims(proto_tensor, axis=1)  # Shape: (num_protos, 1, embd_dim)

    # Calculate Euclidean distances
    distances = keras.ops.sqrt(keras.ops.sum(keras.ops.square(p1 - p2), axis=2))

    # Create a mask for the diagonal (self-distances)
    mask = keras.ops.eye(num_protos) * 1e10

    # Add mask to distances to ignore self-distances
    masked_distances = distances + mask

    # Find minimum distance and corresponding indices
    min_distance = keras.ops.min(masked_distances)

    # Find indices of minimum distance
    min_indices = keras.ops.where(keras.ops.equal(masked_distances, min_distance))[0]
    closest_pair = (0,1)#(int(min_indices[0]), int(min_indices[1]))

    return min_distance, closest_pair, distances

class ProtoLoss(keras.losses.Loss):
    def __init__(self, number_of_persons=5,temperature=1, proto_meaning='users', reduction='sum_over_batch_size',
                 name='CustomLoss'):
        super().__init__(#reduction=reduction,
                         name=name)
        '''proto_meaning could be users or weights'''
        self.number_of_persons = number_of_persons
        self.temperature = temperature
        self.proto_meaning = proto_meaning
        # self.cosine_loss = keras.losses.CosineSimilarity(axis=-1)
        # self.cross_entropy = keras.losses.CategoricalCrossentropy(axis=-1)


    def call(self, y_true, y_pred):
        # Add debugging prints
        has_nans = tf.math.reduce_any(tf.math.is_nan(y_pred))
        if has_nans:
            print('There is Nan values in y_pred')

        if self.proto_meaning == 'weight':
            all_used_weights =sorted(set(y_true.numpy()))
            # value_to_index = {0.0: 0, 0.5: 1, 1.0: 2, 2.0: 3}
            weight_to_index = {weight: i for i,weight in enumerate(all_used_weights)}
            # Convert your float tensor to integers
            y_tr = tf.map_fn(lambda x: int(weight_to_index[float(x)]), y_true)
            y_tr = tf.cast(y_tr, tf.int32)
        else:
            y_tr = tf.cast(y_true, tf.int32)
        one_hot_true = tf.one_hot(y_tr, depth=self.number_of_persons)  # unpacking the predicted output
        partition = tf.dynamic_partition(y_pred, tf.squeeze(y_tr),
                                         num_partitions=self.number_of_persons)
        prototypes = [tf.reduce_mean(x, axis=0) for x in partition]

        min_distance, closest_pair, distances = analyze_prototype_distances(prototypes)
        # batch_size = y_true.shape[0]
        # dist = tf.zeros((batch_size,self.number_of_persons ))
        # for person_num in y_true:
        #     dist[tf.cast(tf.squeeze(tf.where(y_true==person_num), axis=1), dtype=tf.int32),:]=tf.norm(person_num - prototypes[person_num,:], axis=-1)
        # minus_dist = -tf.norm(y_pred - prototypes, axis=-1)#
        # DistanceLayer(prototypes)(y_pred)
        dist = calculate_distances(y_pred, prototypes)/self.temperature
        softmax = tf.nn.softmax(-dist, axis=-1)

        scce = keras.losses.CategoricalCrossentropy()


        loss = scce(one_hot_true, softmax)
        # loss = keras.losses.categorical_crossentropy(one_hot_true, softmax)
        # loss = softmax[0]+y_true[0]
        # Debugging: Print the computed loss
        # print("Loss:", loss)
        # print(f' min dist {min_distance}')
        if tf.math.reduce_any(tf.math.is_nan(min_distance)):
            for prototype in prototypes:
                print(f'prototype {prototype}')

        return loss

    def get_config(self):
        base_config = super().get_config()
        return {
            **base_config,
            "number_of_persons": self.number_of_persons,
            "temperature": self.temperature,
            "proto_meaning": self.proto_meaning,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)