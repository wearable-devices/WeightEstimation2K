import keras
# from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann
import tensorflow as tf
from models import get_optimizer, get_loss
from google_friendly_model.covariances import get_spd_matrices_fixed_point
from google_friendly_model.tangent_proj import TangentSpaceLayer

@keras.utils.register_keras_serializable(package='weight_estimation', name='SequentialCrossSpectralSpd_matricesLayer')
class SequentialCrossSpectralSpd_matricesLayer(tf.keras.layers.Layer):
    def __init__(self, main_window_size=160,
                 # take_tangent_proj=True,
                 metric = 'riemann',
                 est='cov',
                 # base_point_calculation='identity', #'identity' , 'rieman_mean','middle_point', 'first_point'
                 ):
        super(SequentialCrossSpectralSpd_matricesLayer, self).__init__()
        self.main_window_size = main_window_size
        # self.take_tangent_proj = take_tangent_proj
        self.metric = metric
        self.est = est
        # self.base_point_calculation = base_point_calculation
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
        return  csd_matrices#keras.layers.Flatten()(csd_matrices)


    def get_config(self):
        """Return the configuration of the layer for serialization."""
        config = super().get_config()
        config.update({
            'main_window_size': self.main_window_size,
            # 'take_tangent_proj': self.take_tangent_proj,
            'metric': self.metric,
            'est': self.est,
            # 'base_point_calculation': self.base_point_calculation
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
            # 'take_tangent_proj',
           'metric',
        'est',
        # 'base_point_calculation'
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
    spd_matrices_layer = SequentialCrossSpectralSpd_matricesLayer(
        main_window_size=window_size,
        # take_tangent_proj=True,
        # base_point_calculation=base_point_calculation,
    )

    spd_matrices = spd_matrices_layer(sensor_tensor012)

    # Apply tangent space mapping
    tangent_layer = TangentSpaceLayer()#n_channels=3)
    tangent_features = tangent_layer(spd_matrices)

    middle_t_v = tangent_features#tangent_vectors[:,1,:]
    # middle_t_v = keras.layers.Flatten()(spd_matrices)
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


