import tensorflow as tf

class WeightedMeanSquaredError(tf.keras.losses.Loss):
    def __init__(self, weight_dict, reduction=tf.keras.losses.Reduction.AUTO, name="weighted_mean_squared_error"):
        super().__init__(reduction=reduction, name=name)
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
