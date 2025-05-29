import keras
from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann
import tensorflow as tf
from models import get_optimizer, get_loss

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

        return_sequence=False,
        frame_length=frame_length,
        frame_step=frame_step,
        preprocessing_type=preprocessing,
        main_window_size=window_size,  # window size
        take_tangent_proj=True,
        base_point_calculation=base_point_calculation,
    )

    tangent_vectors = csd_layer(sensor_tensor012)
    middle_t_v = tangent_vectors#tangent_vectors[:,1,:]

    # last_tv = tangent_vectors[:,-1,:]
    # dense_layer_1 = keras.layers.Dense(3, activation='linear')#'sigmoid')
    dense_layer_final = keras.layers.Dense(1, activation='sigmoid')

    # x = dense_layer_1(tangent_vectors[:,1,:])
    # x = keras.layers.Dense(1,activation='sigmoid')(x)
    # x = keras.layers.Flatten()(middle_t_v)
    x = keras.layers.Dense(middle_dense_units, activation=dense_activation)(middle_t_v)
    # x = keras.layers.Dense(3, activation='linear')(x)
    # x = keras.layers.Dense(3, activation='linear')(x)
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


