import tensorflow as tf
import numpy as np
from kymatio.scattering1d.filter_bank import scattering_filter_factory
import keras.ops as K
import keras
import matplotlib.pyplot as plt

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

@keras.utils.register_keras_serializable(package='weight_estimation', name='AverageTwoClosest')
class AverageTwoClosest(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Inputs are already (batch_size,) shape
        x1, x2, x3 = inputs[0], inputs[1], inputs[2]

        # Calculate pairwise differences
        diff_12 = K.abs(x1 - x2)
        diff_23 = K.abs(x2 - x3)
        diff_13 = K.abs(x1 - x3)

        # Calculate averages
        avg_12 = (x1 + x2) * 0.5
        avg_23 = (x2 + x3) * 0.5
        avg_13 = (x1 + x3) * 0.5

        # Create masks for the closest pair
        is_12_closest = K.cast(
            K.logical_and(diff_12 <= diff_23, diff_12 <= diff_13),
            dtype='float32'
        )
        is_23_closest = K.cast(
            K.logical_and(diff_23 < diff_12, diff_23 <= diff_13),
            dtype='float32'
        )
        is_13_closest = K.cast(
            K.logical_and(diff_13 < diff_12, diff_13 < diff_23),
            dtype='float32'
        )

        # Compute weighted sum
        result = (avg_12 * is_12_closest +
                  avg_23 * is_23_closest +
                  avg_13 * is_13_closest)

        return result

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Returns (batch_size,)

    def get_config(self):
        config = super().get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[0]  # Return shape of first input (batch_size,)

    @classmethod
    def from_config(cls, config):
        return cls(**config)
# Register the custom layer
keras.saving.get_custom_objects()['AverageTwoClosest'] = AverageTwoClosest

@keras.saving.register_keras_serializable(package='semg_scattering', name='SEMGScatteringTransform')
class SEMGScatteringTransform(keras.layers.Layer):
    def __init__(self,
                 sampling_rate=2000,  # Default sampling rate for sEMG
                 f_powerline=55,  # Power line interference frequencies
                 f_semg=(150, 450),  # Main sEMG band
                 undersampling=1,#support = 128,
                 dc_component=False,
                 **kwargs):
        super(SEMGScatteringTransform, self).__init__(**kwargs)
        self.n_filters_semg = 5
        self.sampling_rate = sampling_rate
        self.f_powerline = f_powerline
        self.f_semg = f_semg
        self.undersampling = undersampling
        self.dc_component = dc_component
        # self.support = support

    def morlet_filter_bank(self, N):
        """
        Generate Morlet wavelet filter bank optimized for sEMG and power line interference
        """
        # Initialize filter banks in frequency domain
        psi1_f = []
        psi2_f = []

        # Calculate frequency bins
        freqs = np.fft.fftfreq(N) * self.sampling_rate

        # First order filters: Split into power line and sEMG regions
        # n_powerline_filters = 1  # Filters for power line interference
        n_semg_filters = self.n_filters_semg

        # Low frequency filters (for motion artifacts and low-frequency components)
        for fc in [10, 30]:  # Low frequency centers at 10Hz and 30Hz
            sigma = 6.0  # Using same Q factor as sEMG filters for consistency
            psi = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
            psi1_f.append({'levels': [psi], 'fc': fc})

        # Power line interference filters (narrower bandwidth)
        # Center frequencies around each power line frequency
        sigma = 5.0  # Narrow bandwidth for power line interference
        psi = np.exp(-0.5 * ((freqs - self.f_powerline) / sigma) ** 2)
        # psi -= np.exp(-0.5 * (freqs / sigma) ** 2)  # Remove DC
        psi1_f.append({'levels': [psi], 'fc': self.f_powerline})

        # sEMG band filters (first order)
        semg_freqs_1 = np.linspace(self.f_semg[0], self.f_semg[1], n_semg_filters)
        for fc in semg_freqs_1:
            sigma = fc / 8.0  # Q factor of 8 for good frequency resolution
            psi = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
            # psi -= np.exp(-0.5 * (freqs / sigma) ** 2)
            psi1_f.append({'levels': [psi], 'fc': fc})

        # High frequency filters
        for fc in [600, 700, 800, 900]:  # High frequency centers
            sigma = fc / 8.0  # Using same Q factor as sEMG filters for consistency
            psi = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
            psi1_f.append({'levels': [psi], 'fc': fc})

        # Second order filters (broader bandwidth, focused on sEMG band)
        semg_freqs_2 = np.linspace(self.f_semg[0], self.f_semg[1], self.n_filters_semg)
        for fc in semg_freqs_2:
            sigma = fc / 4.0  # Q factor of 4 for broader temporal patterns
            psi = np.exp(-0.5 * ((freqs - fc) / sigma) ** 2)
            # psi -= np.exp(-0.5 * (freqs / sigma) ** 2)
            psi2_f.append({'levels': [psi], 'fc': fc})

        # Generate low-pass filter (phi)
        sigma_phi = self.f_semg[1] / 8
        phi_f = np.exp(-0.5 * (freqs / sigma_phi) ** 2)

        return phi_f, psi1_f, psi2_f

    def build(self, input_shape):

        self.filter_len = 128
        # Generate sEMG-optimized filters
        phi_f, psi1_f, psi2_f = self.morlet_filter_bank(self.filter_len)

        # Initialize filter banks
        self.filter_bank_real_l1 = np.zeros((self.filter_len, len(psi1_f)))
        self.filter_bank_imag_l1 = np.zeros((self.filter_len, len(psi1_f)))

        # Convert filters to time domain
        for ii in range(len(psi1_f)):
            time_filter = np.fft.ifft(psi1_f[ii]['levels'][0])
            time_filter = np.fft.ifftshift(time_filter)
            self.filter_bank_real_l1[:, ii] = np.real(time_filter)
            self.filter_bank_imag_l1[:, ii] = np.imag(time_filter)

        # Convert to TensorFlow tensors
        self.filter_bank_real_l1 = tf.convert_to_tensor(self.filter_bank_real_l1, dtype=tf.float32)
        self.filter_bank_imag_l1 = tf.convert_to_tensor(self.filter_bank_imag_l1, dtype=tf.float32)

        # Store dimensions for scattering coefficients
        self.proj0_height = len(psi1_f)
        self.resize_height = self.proj0_height + 1 if self.dc_component else self.proj0_height

        super(SEMGScatteringTransform, self).build(input_shape)



    def call(self, x):
        # First order scattering
        # Calculate real and imaginary responses
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)
        real_response = tf.nn.convolution(
            tf.expand_dims(x, axis=-1),
            tf.expand_dims(self.filter_bank_real_l1, axis=1))
        imag_response = tf.nn.convolution(
            tf.expand_dims(x, axis=-1),
            tf.expand_dims(self.filter_bank_imag_l1, axis=1))


        # Calculate magnitude (as before)
        # magnitude = tf.math.sqrt(tf.math.add(tf.square(real_response), tf.square(imag_response)))
        magnitude = tf.sqrt(tf.maximum(tf.square(real_response) + tf.square(imag_response), 1e-7))

        # Calculate phase
        phase = tf.math.atan2(imag_response, real_response)

        # paddings = tf.constant([[0, 0], [0, 0], [int(self.support / 2), int(self.support / 2)]])
        magnitude_resized = keras.layers.Resizing( int(magnitude.shape[-2]/self.undersampling), self.resize_height)(
            tf.expand_dims(magnitude, axis=-1))

        phase_resized = keras.layers.Resizing( int(magnitude.shape[-2]/self.undersampling), self.resize_height)(
                tf.expand_dims(phase, axis=-1))

        magnitude_resized = tf.squeeze(magnitude_resized, axis=-1)
        phase_resized = tf.squeeze(phase_resized, axis=-1)

        # if self.dc_component:
        #     # DC component (low-pass) - only affects magnitude
        #     low_pass_filter = tf.exp(
        #         -tf.square(tf.range(-self.support // 2, self.support // 2, 1, dtype=tf.float32)) / (
        #                 2 * (self.support // 4) ** 2))
        #     low_pass_filter /= tf.reduce_sum(low_pass_filter)
        #     low_pass_filter = tf.reshape(low_pass_filter, [-1, 1, 1])
        #     dc_component = tf.nn.convolution(tf.expand_dims(x, axis=-1), low_pass_filter, 1, "VALID")
        #     magnitude = tf.concat([dc_component, magnitude], axis=-1)
        #     # Add corresponding zeros for phase of DC component
        #     phase_padding = tf.zeros_like(dc_component)
        #     phase = tf.concat([phase_padding, phase], axis=-1)

        return magnitude_resized, phase_resized

    def plot_filter_bank(self):
        """
        Visualize the filter bank frequency responses with proper frequency scaling
        """
        # Generate filters
        phi_f, psi1_f, psi2_f = self.morlet_filter_bank(self.support)

        # Create frequency axis for plotting (up to Nyquist frequency)
        freqs = np.linspace(0, self.sampling_rate / 2, self.support // 2 + 1)  # Explicit linear spacing to Nyquist
        print(f"Max frequency: {freqs[-1]} Hz")  # Debug print
        print(f"Number of frequency points: {len(freqs)}")  # Debug print

        plt.style.use(['seaborn-v0_8'])
        plt.figure(figsize=(12, 8))

        # Calculate center frequencies for first order filters
        center_freqs_1 = []
        for ii in range(len(psi1_f)):
            phi_i = psi1_f[ii]['levels'][0]
            center_freq = psi1_f[ii]['fc']
            # Take only positive frequencies
            phi_i_plot = phi_i[:len(freqs)]
            # Find center frequency (frequency with maximum magnitude)
            center_freqs_1.append(center_freq)

            plt.subplot(2, 1, 1)
            if center_freq < 55 == 0:  # Only plot once
                plt.plot(freqs, np.abs(phi_f[:len(freqs)]), 'k--',
                         linewidth=2, label='Lowpass Filter $\phi$')
            elif center_freq <= 30:  # Low frequency filters
                plt.plot(freqs, np.abs(phi_i_plot), 'g:',
                         label=f'Lowpass fc={center_freq:.2f}')
            elif center_freq <= self.f_powerline + 5 and center_freq >= self.f_powerline - 5:
                plt.plot(freqs, np.abs(phi_i_plot), '--',
                         label=f'Power Line fc={center_freq:.2f}')
            elif center_freq >= self.f_semg[0] and center_freq <= self.f_semg[1]:
                plt.plot(freqs, np.abs(phi_i_plot),
                         label=f'EMG Filter fc={center_freq:.2f}')
            elif center_freq >= 600:  # High frequency filters
                plt.plot(freqs, np.abs(phi_i_plot), 'r:',
                         label=f'Highpass fc={center_freq:.2f}')
            else:
                plt.plot(freqs, np.abs(phi_i_plot),
                         label=f'Unknown Filter fc={center_freq:.2f}')

        plt.tick_params(labelbottom=True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.title('First Order Morlet basis $\psi^{(1)}$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('$|\psi^{(1)}|$')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(0, self.sampling_rate / 2)
        plt.grid(True)

        # Calculate center frequencies for second order filters
        center_freqs_2 = []
        for ii in range(len(psi2_f)):
            phi_i = psi2_f[ii]['levels'][0]
            phi_i_plot = phi_i[:len(freqs)]
            center_freq = freqs[np.argmax(np.abs(phi_i_plot))]
            center_freqs_2.append(center_freq)

            plt.subplot(2, 1, 2)
            plt.plot(freqs, np.abs(phi_i_plot),
                     label=f'Second Order Filter (fc={center_freq:.0f} Hz)')

        plt.tick_params(labelbottom=True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.title('Second Order Morlet basis $\psi^{(2)}$')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('$|\psi^{(2)}|$')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim(0, self.sampling_rate / 2)
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        return center_freqs_1, center_freqs_2

    def plot_time_domain_filters(self):
        """
        Visualize the time domain representation of the filters
        """
        # Generate filters and get center frequencies
        center_freqs_1, center_freqs_2 = self.plot_filter_bank()
        phi_f, psi1_f, psi2_f = self.morlet_filter_bank(self.N)

        plt.style.use(['seaborn-v0_8'])
        plt.figure(figsize=(12, 8))

        # Plot first order filters
        plt.subplot(2, 1, 1)
        for ii in range(len(psi1_f)):
            phi_i = psi1_f[ii]['levels'][0]
            time_filter = np.fft.ifft(phi_i)
            time_filter = np.fft.ifftshift(time_filter)
            time_axis = np.arange(-len(time_filter) // 2, len(time_filter) // 2) / self.sampling_rate * 1000  # in ms

            if abs(center_freqs_1[ii]) <= max(self.f_powerline) + 10:
                plt.plot(time_axis, np.real(time_filter), '--',
                         label=f'f={abs(center_freqs_1[ii]):.0f} Hz (Power line)')
            else:
                plt.plot(time_axis, np.real(time_filter),
                         label=f'f={abs(center_freqs_1[ii]):.0f} Hz')

        plt.title('First Order Morlet Wavelets (Time Domain)')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        # Plot second order filters
        plt.subplot(2, 1, 2)
        for ii in range(len(psi2_f)):
            phi_i = psi2_f[ii]['levels'][0]
            time_filter = np.fft.ifft(phi_i)
            time_filter = np.fft.ifftshift(time_filter)
            time_axis = np.arange(-len(time_filter) // 2, len(time_filter) // 2) / self.sampling_rate * 1000  # in ms
            plt.plot(time_axis, np.real(time_filter),
                     label=f'f={abs(center_freqs_2[ii]):.0f} Hz')

        plt.title('Second Order Morlet Wavelets (Time Domain)')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def plot_build_filters(self):
        """
        Visualize the time domain filters that are actually used in the build() function
        """
        # Ensure the layer is built
        if not hasattr(self, 'filter_bank_real_l1'):
            raise ValueError("Layer must be built first. Call build() or add to a model before plotting.")

        plt.style.use(['seaborn-v0_8'])
        plt.figure(figsize=(12, 8))

        # Convert filters back to numpy for plotting
        real_filters = self.filter_bank_real_l1.numpy()
        imag_filters = self.filter_bank_imag_l1.numpy()

        # Time axis in milliseconds
        time_axis = np.arange(-self.support // 2, self.support // 2) / self.sampling_rate * 1000

        # Plot real parts
        plt.subplot(2, 1, 1)
        for i in range(real_filters.shape[1]):
            plt.plot(time_axis, real_filters[:, i],
                     label=f'Filter {i + 1} (Real)')

        plt.title('Real Parts of Built Filters')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        # Plot imaginary parts
        plt.subplot(2, 1, 2)
        for i in range(imag_filters.shape[1]):
            plt.plot(time_axis, imag_filters[:, i],
                     label=f'Filter {i + 1} (Imag)')

        plt.title('Imaginary Parts of Built Filters')
        plt.xlabel('Time [ms]')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()


@keras.saving.register_keras_serializable(package='distance_layer', name='DistanceLayer')
class DistanceLayer(keras.layers.Layer):
    def __init__(self, fixed_points, **kwargs):
        super(DistanceLayer, self).__init__(**kwargs)
        self.fixed_points = fixed_points#tf.constant(fixed_points, dtype=tf.float32)

    def call(self, inputs):
        # Calculate distances using broadcasting
        m_distances = -tf.norm(inputs[:, tf.newaxis, :] - self.fixed_points, axis=-1)
        return m_distances

@keras.saving.register_keras_serializable(package='majority_vote', name='MajorityVote')
class MajorityVote(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Expecting a list of 3 inputs, each shape (batch_size,)
        input1, input2, input3 = inputs

        # Stack inputs to shape (batch_size, 3)
        stacked = tf.stack([input1, input2, input3], axis=1)

        # For each sample, sort values
        sorted_vals = tf.sort(stacked, axis=1)

        # Calculate differences between adjacent values
        diff1 = tf.abs(sorted_vals[:, 1] - sorted_vals[:, 0])
        diff2 = tf.abs(sorted_vals[:, 2] - sorted_vals[:, 1])

        # Choose which two values to average based on smallest difference
        result = tf.where(
            diff1 <= diff2,
            # If first two values are closer, average them
            (sorted_vals[:, 0] + sorted_vals[:, 1]) / 2.0,
            # If last two values are closer, average them
            (sorted_vals[:, 1] + sorted_vals[:, 2]) / 2.0
        )

        return result



class GaussianNoiseLayer(keras.layers.Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(shape=tf.shape(inputs),
                                   mean=0.0,
                                   stddev=self.stddev,
                                   dtype=inputs.dtype)
            return inputs + noise
        return inputs

    def get_config(self):
        config = super(GaussianNoiseLayer, self).get_config()
        config.update({'stddev': self.stddev})
        return config