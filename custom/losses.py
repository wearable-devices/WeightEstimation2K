import tensorflow as tf
import keras
import keras.ops as K
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
