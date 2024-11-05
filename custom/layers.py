import tensorflow as tf
import numpy as np
from kymatio.scattering1d.filter_bank import scattering_filter_factory
import keras.ops as K
import keras

# @keras.saving.register_keras_serializable(package='weight_estimation', name='ScatteringTimeDomain')
@keras.utils.register_keras_serializable(package='weight_estimation', name='ScatteringTimeDomain')

class ScatteringTimeDomain(keras.layers.Layer):
    # Please help me implement the second layer as you suggested
    def __init__(self, J, Q, undersampling, max_order, support=None, sigma_phi=None, dc_component=False, **kwargs):
        super(ScatteringTimeDomain, self).__init__(**kwargs)
        self.J = J
        self.Q = Q
        self.undersampling = undersampling
        self.max_order = max_order

        # If sigma is not provided, set it as a quarter of the support by default
        self.support = 2 ** self.J if support is None else support
        self.sigma_phi = sigma_phi if sigma_phi is not None else self.support // 4
        self.dc_component = dc_component

    def total_combinations_second_layer(self, J, Q):
        n = J * Q[1] - 1
        total = 0
        for k in range(1, n + 1):
            total += k * (k + 1) // 2
        return total

    def build(self, input_shape):
        self.trainable = False
        self.N = 256  # Scattering filter need a number 2^x for dyadic subsampling

        # Generate filters - This implementation is partial and hard-coded to avoid conversion issues
        phi_f, psi1_f, psi2_f = scattering_filter_factory(self.N, self.J, self.Q, self.support)

        self.filter_bank_real_l1 = np.zeros((self.support, self.J * self.Q[0]))
        self.filter_bank_imag_l1 = np.zeros((self.support, self.J * self.Q[0]))

        self.filter_bank_real_l2 = np.zeros((self.support, self.J * self.Q[1]))
        self.filter_bank_imag_l2 = np.zeros((self.support, self.J * self.Q[1]))

        self.proj0_height = len(psi1_f)
        self.proj1_height = self.total_combinations_second_layer(self.J, self.Q)
        self.resize_height = self.proj0_height + 1 if self.dc_component else self.proj0_height

        # Note - removed first filter (there are J+1 filters, due to the 'DC')
        # Todo - fix center point (with odd convolution length?)
        for ii in range(0, len(psi1_f)):
            self.filter_bank_real_l1[:, ii - 1] = np.fft.ifftshift(np.real(np.fft.ifft(psi1_f[ii]['levels'][0])))[
                                                  int(self.N / 2) - int(self.support / 2):int(self.N / 2) + int(
                                                      self.support / 2)]
            self.filter_bank_imag_l1[:, ii - 1] = np.fft.ifftshift(np.imag(np.fft.ifft(psi1_f[ii]['levels'][0])))[
                                                  int(self.N / 2) - int(self.support / 2):int(self.N / 2) + int(
                                                      self.support / 2)]

        for ii in range(0, len(psi2_f)):
            self.filter_bank_real_l2[:, ii - 1] = np.fft.ifftshift(np.real(np.fft.ifft(psi2_f[ii]['levels'][0])))[
                                                  int(self.N / 2) - int(self.support / 2):int(self.N / 2) + int(
                                                      self.support / 2)]
            self.filter_bank_imag_l2[:, ii - 1] = np.fft.ifftshift(np.imag(np.fft.ifft(psi2_f[ii]['levels'][0])))[
                                                  int(self.N / 2) - int(self.support / 2):int(self.N / 2) + int(
                                                      self.support / 2)]

        self.filter_bank_real_l1 = tf.expand_dims(tf.convert_to_tensor(self.filter_bank_real_l1, dtype=tf.float32),
                                                  axis=1)
        self.filter_bank_imag_l1 = tf.expand_dims(tf.convert_to_tensor(self.filter_bank_imag_l1, dtype=tf.float32),
                                                  axis=1)
        self.filter_bank_real_l2 = tf.expand_dims(tf.convert_to_tensor(self.filter_bank_real_l2, dtype=tf.float32),
                                                  axis=1)
        self.filter_bank_imag_l2 = tf.expand_dims(tf.convert_to_tensor(self.filter_bank_imag_l2, dtype=tf.float32),
                                                  axis=1)

        super(ScatteringTimeDomain, self).build(input_shape)

    def call(self, x):

        proj0_real = tf.square(
            tf.nn.convolution(tf.expand_dims(x, axis=-1), self.filter_bank_real_l1, 1, "VALID"))
        proj0_imag = tf.square(
            tf.nn.convolution(tf.expand_dims(x, axis=-1), self.filter_bank_imag_l1, 1, "VALID"))
        proj0 = tf.math.sqrt(tf.math.add(proj0_real, proj0_imag))

        if self.dc_component:
            # DC component (low-pass)
            low_pass_filter = tf.exp(
                -tf.square(tf.range(-self.support // 2, self.support // 2, 1, dtype=tf.float32)) / (
                        2 * self.sigma_phi ** 2))
            # Normalize the filter
            low_pass_filter /= tf.reduce_sum(low_pass_filter)
            low_pass_filter = tf.reshape(low_pass_filter, [-1, 1, 1])
            dc_component = tf.nn.convolution(tf.expand_dims(x, axis=-1), low_pass_filter, 1, "VALID")

            # Append the DC component to the first order coefficients
            proj0 = tf.concat([dc_component, proj0], axis=-1)

            # If max order is 1, just return the first order coefficients
        # resize_height = self.proj0_height + 1 if self.dc_component else self.proj0_height
        if self.max_order == 1:
            return keras.layers.Resizing(self.resize_height, int(self.N / self.undersampling))(
                tf.expand_dims(tf.transpose(proj0, perm=(0, 2, 1)), axis=-1))

        else:
            second_order_coeffs = []
            for j1 in range(self.J * self.Q[0]):
                for j2 in range(j1 + 1, self.J * self.Q[1]):
                    # print('j1 = '  +str(j1) + ', j2 = ' + str(j2))
                    proj1_real = tf.square(
                        tf.nn.convolution(proj0[..., j1:j1 + 1], self.filter_bank_real_l2[..., j2:j2 + 1], 1, "VALID"))
                    proj1_imag = tf.square(
                        tf.nn.convolution(proj0[..., j1:j1 + 1], self.filter_bank_imag_l2[..., j2:j2 + 1], 1, "VALID"))
                    proj1 = tf.math.sqrt(tf.math.add(proj1_real, proj1_imag))
                    second_order_coeffs.append(proj1)

            proj1 = tf.concat(second_order_coeffs, axis=-1)

            # Define padding for 4D tensor
            # [
            #   [pad_before_dim_1, pad_after_dim_1],
            #   [pad_before_dim_2, pad_after_dim_2],
            #   [pad_before_dim_3, pad_after_dim_3],
            # ]
            # In this case, we want to pad only the width dimension (dim_3).
            paddings = tf.constant([[0, 0], [0, 0], [int(self.support / 2), int(self.support / 2)]])

            # Resize the second-order coefficients and the first-order coefficients
            proj0_resized = keras.layers.Resizing(self.resize_height, int(self.N / self.undersampling))(
                tf.expand_dims(tf.transpose(proj0, perm=(0, 2, 1)), axis=-1))
            proj1_resized = keras.layers.Resizing(self.proj1_height, int(self.N / self.undersampling))(
                tf.expand_dims(tf.pad(tf.transpose(proj1, perm=(0, 2, 1)), paddings, "SYMMETRIC"), axis=-1))
            # proj1_resized = tf.pad(proj1_resized, paddings, "SYMMETRIC")

            return proj0_resized, proj1_resized

    def get_config(self):
        config = super(ScatteringTimeDomain, self).get_config()
        config.update({
            'J': self.J,
            'Q': self.Q,
            'undersampling': self.undersampling,
            'max_order': self.max_order,
            'support': self.support,
            'sigma_phi': self.sigma_phi,
            'dc_component': self.dc_component
        })
        return config


# Register the custom layer for loading
keras.utils.get_custom_objects()['ScatteringTimeDomain'] = ScatteringTimeDomain




@keras.utils.register_keras_serializable(package='weight_estimation', name='CustomMultiHeadAttention')
class CustomMultiHeadAttention(keras.layers.Layer):
    def __init__(self, key_dim, num_heads=1, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

    def build(self, input_shape):
        self.scale = self.key_dim ** -0.5
        self.query_dense = keras.layers.Dense(units=self.key_dim, activation='linear')
        self.key_dense = keras.layers.Dense(units=self.key_dim, activation='linear')

    def call(self, query, key, value, return_attention_scores=True):
        embd_query = self.query_dense(query)
        embd_key = self.key_dense(key)

        # Calculate attention scores
        matmul_qk = tf.matmul(embd_query, embd_key, transpose_b=True)
        scaled_attention_logits = matmul_qk * self.scale

        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

        # Apply attention weights to values
        output = tf.matmul(attention_weights, value)

        return output, attention_weights

    def get_config(self):
        config = super(CustomMultiHeadAttention, self).get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ScaleLayer(keras.layers.Layer):
    def __init__(self, scale=0.1, **kwargs):
        super(ScaleLayer, self).__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale})
        return config


@keras.utils.register_keras_serializable(package='weight_estimation', name='CustomPositionalEmbedding')

class CustomPositionalEmbedding(keras.layers.Layer):
    def __init__(self, scale_activation='linear', **kwargs):
        super().__init__(**kwargs)
        if scale_activation == 'mult01':
            self.scale_activation = ScaleLayer(scale=0.1)
        else:
            self.scale_activation = keras.layers.Activation(scale_activation)

    def call(self, inputs):
        # inputs shape: (batch_size, time_steps, n)
        batch_size, time_steps, n = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        # Create position encoding
        position_encoding = self.scale_activation(tf.range(time_steps, dtype=tf.float32) / tf.cast(time_steps, tf.float32))
        position_encoding = tf.expand_dims(tf.expand_dims(position_encoding, 0), -1)

        # Tile position encoding for all batches
        position_encoding = tf.tile(position_encoding, [batch_size, 1, 1])

        # Concatenate original input with position encoding
        return tf.concat([inputs, position_encoding], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + 1,)

    def get_config(self):
        config = super().get_config()
        config.update({"scale_activation": self.scale_activation})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

@keras.utils.register_keras_serializable(package='weight_estimation', name='OrderedAttention')
class OrderedAttention(keras.layers.Layer):
    def __init__(self, num_heads, key_dim, scale_activation='linear', name=None, **kwargs):
        super(OrderedAttention, self).__init__(name=name, **kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.scale_activation = scale_activation
        # Don't create layers in __init__, do it in build()
        self.attention_layer = None
        self.positional_embedding = None

    def build(self, input_shape):
        self.attention_layer = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim,
            name=f"{self.name}_mha" if self.name else None
        )
        self.positional_embedding = CustomPositionalEmbedding(
            scale_activation=self.scale_activation,
            name=f"{self.name}_pos_emb" if self.name else None
        )
        super().build(input_shape)

    def call(self, query, value, return_attention_scores=False):
        query_with_position = self.positional_embedding(query)
        value_with_position = self.positional_embedding(value)
        return self.attention_layer(
            query_with_position,
            value_with_position,
            return_attention_scores=return_attention_scores
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "scale_activation": self.scale_activation,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def sensor_image_layers(sensor_1_image, units, activation, apply_tfp=False, apply_noise=True,stddev=0.1, sensor_num=1):

    x_mean = K.mean(sensor_1_image, axis=1)
    x_min = K.min(sensor_1_image, axis=1)
    x_max =K.max(sensor_1_image, axis=1)
    new_image = K.concatenate([K.expand_dims(x_min, axis=2), K.expand_dims(x_mean, axis=2),
                           K.expand_dims(x_max, axis=2)], axis=2)

    if apply_noise:
        noise_memory = keras.backend.in_train_phase(
            add_noise(new_image, stddev=stddev),
            new_image)
    if apply_tfp:
        # tfp.layers.DenseVariational(units=20, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn)
        x = tfp.layers.DenseVariational(units=20, make_prior_fn=prior_fn, make_posterior_fn=posterior_fn, activation=activation)(new_image)
        x = keras.layers.Flatten()(x)
        x = tfp.layers.DenseVariational(units=units, activation=activation, name=f'dense_1_for_sensor_{sensor_num}')(x)
        x = keras.layers.Dropout(0.1)(x)
        x = tfp.layers.DenseVariational(units=units // 2, activation=activation)(x)  # , name=f'dense_2_for_sensor_{sensor_num}')(x)
    else:
        x = keras.layers.Dense(1, activation=activation)(new_image)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(units, activation=activation, name=f'dense_1_for_sensor_{sensor_num}')(x)
        x = keras.layers.Dropout(0.1)(x)

    return x