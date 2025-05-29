import keras
import tensorflow as tf
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from custom.layers import ScatteringTimeDomain



# import tensorflow as tf
# from pyriemann.estimation import Covariances
# from pyriemann.tangentspace import TangentSpace
#
@keras.utils.register_keras_serializable(package='weight_estimation', name='SequentialCrossSpectralDensityLayer_pyriemann')
class SequentialCrossSpectralDensityLayer_pyriemann(tf.keras.layers.Layer):
    def __init__(self, main_window_size=160,
                 take_tangent_proj=True,
                 metric = 'riemann',
                 preprocessing_type='scattering',
                 return_sequence=False,
                 frame_length=90,
                 frame_step = 8,
                 est='cov',
                 base_point_calculation='identity', #'identity' , 'rieman_mean','middle_point', 'first_point'
                 ):
        super(SequentialCrossSpectralDensityLayer_pyriemann, self).__init__()
        self.return_sequence = return_sequence
        self.frame_length = frame_length # used only in case of return_sequence
        self.frame_step = frame_step # used only in case of return_sequence
        self.main_window_size = main_window_size
        self.take_tangent_proj = take_tangent_proj
        self.metric = metric
        self.preprocessing_type = preprocessing_type
        self.est = est
        self.base_point_calculation = base_point_calculation
        self._n_channels = None  # Will be set in build method

    def build(self, input_shape):
        # Set the number of channels during build
        self._n_channels = input_shape[-2]

        # Mark the layer as built
        super().build(input_shape)

    def preprocessing(self, x):
        return  preprocessing(x, self.preprocessing_type)#,J_snc=self.J_snc, Q_snc=self.Q_snc, undersampling=self.undersampling,
                  # fft_window_size=self.fft_window_size, fft_stride=self.fft_stride, _window=self._window,
                  # segments_to_average=self.segments_to_average, segments_stride=self.segments_stride)

    def compute_csd_matrices(self, x):
        """Compute Cross-Spectral Density matrices using Welch's method.
        x (batch_size, channels, window_size)"""

        if self.preprocessing_type is not None:
            x = self.preprocessing(x)
            # x = tf.reduce_mean(x, axis=-2)
            # x = tf.transpose(x, perm=(0,2,1))
            # Take average for freq bins
            x = tf.concat([tf.reduce_mean(x[:, :4,...], axis=1, keepdims=True), tf.reduce_mean(x[:, 4:8,...], axis=1, keepdims=True),
                       tf.reduce_mean(x[:, 8:,...], axis=1, keepdims=True)], axis=1) # (batch_size, 3,ch,1)
            x = tf.transpose(x, perm=(0,1,3,2))
        def pyriemann_cov(inputs, est):
            import pyriemann.utils.covariance
            return pyriemann.utils.covariance.covariances(inputs.numpy(), estimator=est)

        conc_freq_bins = x# tf.expand_dims(x, axis=1)

        if self.return_sequence:
            framed_conc_freq_bins = tf.signal.frame(
                conc_freq_bins,
                frame_length=self.frame_length,
                frame_step=self.frame_step,
                axis=-1
            )
            # Get shape information
            batch_size = tf.shape(framed_conc_freq_bins)[0]
            n_channels= framed_conc_freq_bins.shape[1]
            seq_len = framed_conc_freq_bins.shape[2]
            # subwin_size = framed_conc_freq_bins.shape[3] #=frame_length

            # Reshape to merge batch, freq, seq dimensions for processing
            merged_shape = (-1, n_channels, self.frame_length)  # Combine batch, freq, seq
            reshaped = tf.reshape(framed_conc_freq_bins, merged_shape)


            csd_raw = tf.py_function(
                func=lambda x: pyriemann_cov(x, self.est),#pyriemann_cov,
                inp=[reshaped],
                Tout=tf.float32
            )

            # Reshape back to separate batch, freq, seq dimensions
            csd_matrices = tf.reshape(csd_raw, (batch_size, seq_len, n_channels, n_channels))
        else:
            if self.preprocessing_type is None:
                csd_matrices = tf.py_function(
                    func=lambda x: pyriemann_cov(x, self.est),
                    inp=[x],
                    Tout=tf.float32
                )
            else:
                # Get shape information
                batch_size = tf.shape(x)[0]
                n_channels = x.shape[-2]
                freq_bin = x.shape[1]
                time_len = x.shape[-1]

                # Reshape to merge batch, freq, seq dimensions for processing
                merged_shape = (-1, n_channels, time_len)  # Combine batch, freq, seq
                reshaped = tf.reshape(x, merged_shape)

                csd_raw = tf.py_function(
                    func=lambda x: pyriemann_cov(x, self.est),  # pyriemann_cov,
                    inp=[reshaped],
                    Tout=tf.float32
                )

                # Reshape back to separate batch, freq, seq dimensions
                csd_matrices = tf.reshape(csd_raw, (batch_size, freq_bin, n_channels, n_channels))
        return csd_matrices#tf.stack(csd_matrices, axis=1)

    def call(self, inputs):
        # Compute CSD directly on input
        csd_matrices = self.compute_csd_matrices(inputs) # (batch_size,freq_bin, ch,ch)
        if not self.take_tangent_proj:
            # flat last two dim
            tril = tf.linalg.band_part(csd_matrices, -1, 0) - tf.linalg.diag(tf.linalg.diag_part(csd_matrices))
            tril_unique = extract_strictly_lower_triangular_flattened(tril)
            diag_part = tf.linalg.diag_part(csd_matrices)
            out = tf.concat([diag_part, tril_unique], axis=-1)
            return out
        else:
            ts = TangentSpace(metric=self.metric)

            def pyriemann_ts(inputs):
                import pyriemann.utils.covariance
                return ts.transform(inputs.numpy())
            # Project to tangent space
            if self.return_sequence or self.preprocessing_type is not None:


                # Get shape information
                batch_size = tf.shape(csd_matrices)[0]
                n_channels = csd_matrices.shape[2]
                seq_len = csd_matrices.shape[1]

                # Reshape to merge batch, freq, seq dimensions for processing
                merged_shape = (-1, n_channels, n_channels)  # Combine batch, freq, seq
                reshaped = tf.reshape(csd_matrices, merged_shape)

                csd_raw = tf.py_function(
                    func=lambda x: pyriemann_ts(x),
                    inp=[reshaped],
                    Tout=tf.float32
                )

                # IMPORTANT: Set the shape explicitly after py_function
                output_dim = n_channels * (n_channels + 1) // 2
                csd_raw.set_shape([None, output_dim])

                # Reshape back to separate batch, freq, seq dimensions
                tangent_vectors = tf.reshape(csd_raw, (batch_size, seq_len, output_dim))

            else:
                if self.preprocessing_type is None:
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
            'preprocessing_type':self.preprocessing_type,
            'metric': self.metric,
            'return_sequence': self.return_sequence,
            'frame_length': self.frame_length,
            'frame_step': self.frame_step,
            'est': self.est,
            'base_point_calculation': self.base_point_calculation
        })
        return config

    # @classmethod
    # def from_config(cls, config):
    #     """Create a new instance from the serialized configuration."""
    #     # Define explicitly which config keys should be passed to the constructor
    #     allowed_kwargs = [
    #         'main_window_size',
    #         'take_tangent_proj',
    #         'metric',
    #         'return_sequence',
    #         'frame_length',
    #         'frame_step',
    #         'est',
    #         'base_point_calculation'
    #     ]
    #
    #     # These keys are special and should not be passed directly to __init__
    #     # They are handled by the parent Layer class
    #     special_kwargs = ['name', 'trainable', 'dtype', 'dynamic']
    #
    #     # Create filtered config with only allowed parameters
    #     filtered_config = {}
    #
    #     # First add the parameters we know our class accepts
    #     for kwarg in allowed_kwargs:
    #         if kwarg in config:
    #             filtered_config[kwarg] = config[kwarg]
    #
    #     # Then add special parameters that should be passed through to super().__init__
    #     for kwarg in special_kwargs:
    #         if kwarg in config:
    #             filtered_config[kwarg] = config[kwarg]
    #
    #     return cls(**filtered_config)

    @classmethod
    def from_config(cls, config):
        """Create a layer from its config.

        This method is used when loading a layer from a saved model.
        """
        # Filter out Keras-specific parameters that aren't used by your layer
        layer_config = {k: v for k, v in config.items() if k in [
            'main_window_size',
            'preprocessing_type',
            'fft_window_size',
            'fft_stride',
            'segments_to_average',
            'segments_stride',
            'base_point_calculation',
            'freq_bins',
            'fs', 'return_sequence'
        ]}
        return cls(**layer_config)

def preprocessing(signal, preprocessing_type='scattering', J_snc=5, Q_snc=(2, 1), undersampling=4,
                  fft_window_size=32, fft_stride=16, _window=None,
                  segments_to_average=16, segments_stride=8):
    '''x of shape (batch_size, window_size, ch=3)'''
    if preprocessing_type == 'scattering':
        scattered_snc = []
        for i in range(signal.shape[1]):
            scattered_snc1, scattered_snc11 = ScatteringTimeDomain(J=J_snc, Q=Q_snc,
                                                                   undersampling=undersampling,
                                                                   max_order=2)(signal[:, i, :])
            scattered_snc.append(scattered_snc1)
        scatt_time_sequence = tf.concat(scattered_snc, axis=-1)  # (batch_size,freqs,time,channels)

        return scatt_time_sequence







import pandas as pd
def get_framed_snc_data_from_file(file_path, window_size=64, frame_step=16):
    data = pd.read_csv(file_path)

    # Extract sensors
    snc1 = data['Snc1'].values
    snc2 = data['Snc2'].values
    snc3 = data['Snc3'].values

    signal = tf.concat([tf.expand_dims(tf.convert_to_tensor(snc1), axis=-1),
                        tf.expand_dims(tf.convert_to_tensor(snc2), axis=-1),
                        tf.expand_dims(tf.convert_to_tensor(snc3), axis=-1)], axis=-1)

    framed_signal = tf.signal.frame(signal, window_size,
                                    frame_step=frame_step,
                                    pad_end=False,
                                    pad_value=0,
                                    axis=-2,
                                    name=None
                                    )

    return framed_signal, snc1, snc2, snc3#, snc_button
if __name__ == "__main__":
    window_size= 648
    frame_step= 18
    train_file = '/home/wld-algo-6/Data/SortedCleaned/Lee/press_release/Lee_1_press_0_Nominal_TableTop_M.csv'


    # get labeled data from csv filles
    framed_signal, snc1, snc2, snc3 = get_framed_snc_data_from_file(train_file, window_size=window_size, frame_step=frame_step)

    # Create CSD Layer
    csd_layer = SequentialCrossSpectralDensityLayer_pyriemann(

        return_sequence=False,
        frame_length=frame_step,
        frame_step=frame_step,
        preprocessing_type='scattering',
        main_window_size=window_size,  # window size
        take_tangent_proj=True,
        base_point_calculation='identity',
    )

    tangent_vectors = csd_layer(tf.transpose(framed_signal, perm=(0,2,1)))

    ttt =1
