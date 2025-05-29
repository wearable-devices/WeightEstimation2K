import keras
# from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann
import tensorflow as tf
from models import get_optimizer, get_loss
from google_friendly_model.covariances import get_spd_matrices_fixed_point

@keras.utils.register_keras_serializable(package='weight_estimation', name='SequentialCrossSpectralDensityLayer_pyriemann')
class SequentialCrossSpectralDensityLayer_pyriemann(tf.keras.layers.Layer):
    def __init__(self, main_window_size=160,
                 take_tangent_proj=True,
                 metric = 'riemann',
                 # preprocessing_type='scattering',
                 # return_sequence=False,
                 # frame_length=90,
                 # frame_step = 8,
                 est='cov',
                 base_point_calculation='identity', #'identity' , 'rieman_mean','middle_point', 'first_point'
                 ):
        super(SequentialCrossSpectralDensityLayer_pyriemann, self).__init__()
        self.main_window_size = main_window_size
        self.take_tangent_proj = take_tangent_proj
        self.metric = metric
        self.est = est
        self.base_point_calculation = base_point_calculation
        self._n_channels = None  # Will be set in build method

    def build(self, input_shape):
        # Set the number of channels during build
        self._n_channels = input_shape[-2]

        # Mark the layer as built
        super().build(input_shape)

    def compute_csd_matrices(self, x):
        """Compute Cross-Spectral Density matrices using Welch's method.
        x (batch_size, channels, window_size)"""

        csd_matrices = get_spd_matrices_fixed_point(x)

        return csd_matrices#tf.stack(csd_matrices, axis=1)

    def call(self, inputs):
        # Compute CSD directly on input
        csd_matrices = self.compute_csd_matrices(inputs) # (batch_size,freq_bin, ch,ch)
        if not self.take_tangent_proj:
            # flat last two dim
            # tril = tf.linalg.band_part(csd_matrices, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(csd_matrices))
            # tril_unique = extract_strictly_lower_triangular_flattened(tril)
            # diag_part = tf.linalg.diag_part(csd_matrices)
            # out = tf.concat([diag_part, tril_unique], axis=-1)
            # return out
            return keras.layers.Flatten()(csd_matrices)
        else:
            ts = TangentSpace(metric=self.metric)

            def pyriemann_ts(inputs):
                import pyriemann.utils.covariance
                return ts.transform(inputs.numpy())
            # Project to tangent space

            # tangent_vectors = ts.transform(csd_matrices.numpy())
            tangent_vectors = tf.py_function(
                func=lambda x: pyriemann_ts(x),
                inp=[csd_matrices],
                Tout=tf.float32
            )
            # IMPORTANT: Calculate and set the expected output shape
            output_dim = self._n_channels * (self._n_channels + 1) // 2
            tangent_vectors.set_shape([None, output_dim])

            return tangent_vectors
        # return csd_matrices

    def get_config(self):
        """Return the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'main_window_size': self.main_window_size,
            'take_tangent_proj': self.take_tangent_proj,
            # 'preprocessing_type':self.preprocessing_type,
            'metric': self.metric,
            # 'return_sequence': self.return_sequence,
            # 'frame_length': self.frame_length,
            # 'frame_step': self.frame_step,
            'est': self.est,
            'base_point_calculation': self.base_point_calculation
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create a layer from its config.

        This method is used when loading a layer from a saved model.
        """
        # Filter out Keras-specific parameters that aren't used by your layer
        layer_config = {k: v for k, v in config.items() if k in [
            'main_window_size',
            # 'preprocessing_type',
            # 'fft_window_size',
            # 'fft_stride',
            # 'segments_to_average',
            # 'segments_stride',
            'base_point_calculation',
            # 'freq_bins',
            # 'fs', 'return_sequence'
        ]}
        return cls(**layer_config)



def mpf_model(window_size=648,base_point_calculation='identity',
              frame_length = 90, frame_step=8, preprocessing=None,
    middle_dense_units = 3,dense_activation='linear',
              max_weight = 2,
            optimizer='Adam', learning_rate=0.0016,
                            loss = 'Huber',
              compile=True
              ):

    '''
    nn_type='attention' could be 'attention' or 'lstm',
    use_positional_encoding_for_timesteps --- use OrderedAttention instead of MultiHeadAttention for scattered snc
    '''


    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=window_size, name='snc_1')
    input_layer_snc2 = keras.Input(shape=window_size, name='snc_2')
    input_layer_snc3 = keras.Input(shape=window_size, name='snc_3')

    sensor_tensor012 = tf.stack([input_layer_snc1, input_layer_snc2, input_layer_snc3], axis=-1)
    sensor_tensor012 = tf.transpose(sensor_tensor012, perm=(0, 2, 1))

    # Create CSD Layer
    csd_layer = SequentialCrossSpectralDensityLayer_pyriemann(
        main_window_size=window_size,  # window size
        take_tangent_proj=False,#True,
        base_point_calculation=base_point_calculation,
    )

    tangent_vectors = csd_layer(sensor_tensor012)
    middle_t_v = tangent_vectors#tangent_vectors[:,1,:]
    middle_t_v = keras.layers.Flatten()(middle_t_v)
    dense_layer_final = keras.layers.Dense(1, activation='sigmoid')
    x = keras.layers.Dense(middle_dense_units, activation=dense_activation)(middle_t_v)
    output = max_weight * dense_layer_final(x)

    model = keras.Model(inputs=[input_layer_snc1, input_layer_snc2, input_layer_snc3],
                        outputs=output,  # [fused_out, snc_output_1, snc_output_2, snc_output_3],
                        name='psd_weight_estimation_model')


    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
        loss = get_loss(loss)
        model.compile(
            optimizer=opt,
            loss=loss,
            metrics=['mae'],

        )

    return model


