import tensorflow as tf
from custom.layers import *
import keras.ops as K

def get_optimizer(optimizer = 'LAMB',learning_rate=0.001, weight_decay=0.0):
    # https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
    # if optimizer == 'LAMB':
    #     opt = tfa.optimizers.LAMB(learning_rate=learning_rate, beta_1=0.9, weight_decay=weight_decay)
    if optimizer == 'Adam':
        opt = tf.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer == 'SGD':
        opt = tf.optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = optimizer

    return opt


def sensor_attention_processing(scattered_snc, units = 5, conv_activation='tanh',
                                use_attention=True, apply_tfp=False,
                                num_heads=5, key_dim=10, attention_layers_for_one_sensor=1,  apply_noise=True,
                                    stddev=0.01,
                                use_time_ordering=False,scale_activation='linear', sensor_num=1):
    if use_attention:
        if use_time_ordering:
            time_attention_layer_1 = OrderedAttention(num_heads=num_heads, key_dim=key_dim, scale_activation=scale_activation, name=f'time_attention1_for_sensor_{sensor_num}')
            time_attention_layer_2 = OrderedAttention(num_heads=num_heads, key_dim=key_dim,
                                                      scale_activation=scale_activation,
                                                      name=f'time_attention2_for_sensor_{sensor_num}')
        else:
            time_attention_layer_1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                                                      name=f'time_attention1_for_sensor_{sensor_num}')
            time_attention_layer_2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                                                        name=f'time_attention2_for_sensor_{sensor_num}')
        attended, _ = time_attention_layer_1(scattered_snc, scattered_snc, return_attention_scores=True)
        if attention_layers_for_one_sensor >1:
            attended, _ = time_attention_layer_2(attended, attended, return_attention_scores=True)
    else:
        attended = scattered_snc

    x = sensor_image_layers(attended, units=units, activation=conv_activation, sensor_num=sensor_num, apply_tfp=apply_tfp,
                                apply_noise=apply_noise, stddev=stddev)
    return  x

def create_attention_weight_estimation_model(window_size_snc=306, apply_tfp=False,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             units=10, conv_activation='tanh', use_attention=True,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             use_sensor_attention=False,
                                             use_sensor_ordering=False,
                                             final_activation='sigmoid',
                                             num_heads=4, key_dim_for_snc=4, key_dim_for_sensor_att=4,
                                             scale_activation='linear',
                                             num_sensor_attention_heads=2,
                                             use_probabilistic_app=False,
                                             prob_param={'smpl_rate': 49,  # trial.suggest_int('smpl_rate', 8, 50),
                                                         'sigma_for_labels_prob': 0.4, },
                                             apply_noise=True, stddev=0.1,
                                             optimizer='LAMB', learning_rate=0.0016,
                                             weight_decay=0.0, max_weight=8, compile=True,
                                             use_weighted_loss=True, normalization_factor=3,
                                             weight_loss_multipliers_dict={weight: 1 for weight in [0, 1, 2, 4, 6, 8]}
                                             ):
    # Define inputs to the model
    input_layer_snc1 = tf.keras.Input(shape=(window_size_snc,), name='Snc1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = tf.keras.Input(shape=(window_size_snc,), name='Snc2')
    input_layer_snc3 = tf.keras.Input(shape=(window_size_snc,), name='Snc3')

    scattered_snc1, scattered_snc11 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc1)
    scattered_snc2, scattered_snc22 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc2)
    scattered_snc3, scattered_snc33 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc3)

    scaterred_snc_list = [  K.squeeze(scattered_snc1, axis=-1),
                          K.squeeze(scattered_snc2, axis=-1),
                          K.squeeze(scattered_snc3, axis=-1)]
    scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                            K.squeeze(scattered_snc22, axis=-1),
                            K.squeeze(scattered_snc33, axis=-1)]

    S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
    S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))

    S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
    S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))

    S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
    S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))

    S_snc1 = S_snc1[:, :, :20]
    S_snc2 = S_snc2[:, :, :20]
    S_snc3 = S_snc3[:, :, :20]

    # Apply Time attention
    sensor_1 = sensor_attention_processing(S_snc1, units=units, conv_activation=conv_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering, apply_tfp=apply_tfp,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=1, use_attention=use_attention,)
    sensor_2 = sensor_attention_processing(S_snc2, units=units, conv_activation=conv_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering, apply_tfp=apply_tfp,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=2, use_attention=use_attention)
    sensor_3 = sensor_attention_processing(S_snc3, units=units, conv_activation=conv_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering, apply_tfp=apply_tfp,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=3, use_attention=use_attention)

    x1 = sensor_1
    x2 = sensor_2
    x3 = sensor_3

    if use_sensor_attention:
        sensor_conc = K.concatenate([K.expand_dims(x1, axis=1), K.expand_dims(x2, axis=1), K.expand_dims(x3, axis=1)],
                                axis=1)
        if use_sensor_ordering:
            sensor_attention_layer = OrderedAttention(num_heads=num_sensor_attention_heads,
                                                      key_dim=key_dim_for_sensor_att, scale_activation=scale_activation)
        else:
            sensor_attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=key_dim_for_sensor_att)

        attended, _ = sensor_attention_layer(sensor_conc, sensor_conc, return_attention_scores=True)
        x1 = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name='sensor_1_attended')(attended)
        x2 = tf.keras.layers.Lambda(lambda x: x[:, 1, :], name='sensor_2_attended')(attended)
        x3 = tf.keras.layers.Lambda(lambda x: x[:, 2, :], name='sensor_3_attended')(attended)

        mean = K.mean(attended, axis=1)
    else:

        x1 = tf.keras.layers.Dense(units, activation='softmax')(x1)
        x2 = tf.keras.layers.Dense(units, activation='softmax')(x2)
        x3 = tf.keras.layers.Dense(units, activation='softmax')(x3)

    if use_probabilistic_app:
        smpl_rate = prob_param['smpl_rate']
        sigma_for_labels_prob = prob_param['sigma_for_labels_prob']
        B = tf.shape(attended)[0]  # Get the batch size from attended tensor
        weights = tf.constant([0, 1, 2, 4, 6, 8])
        key = create_vector_tensor([0, 1, 2, 4, 6, 8], B, max_value=8)
        # key = transform_tensor(tf.expand_dims(weights,axis=1), max_weight)
        prob_support_labels = transform_tensor_with_gaussian(tf.expand_dims(weights, axis=1), smpl_rate, max_weight,
                                                             sigma=sigma_for_labels_prob)
        prob_support_labels = tf.squeeze(prob_support_labels, axis=1)
        main_attention_layer = CustomMultiHeadAttention(key_dim=key_dim_for_snc, num_heads=3)

        # Reshape and tile prob_support_labels to match the batch size
        prob_support_labels_reshaped = tf.tile(
            tf.expand_dims(prob_support_labels, axis=0),  # Add batch dimension
            [B, 1, 1]  # Repeat for each item in the batch
        )

        # Ensure the shape is correct
        # prob_support_labels_reshaped = tf.ensure_shape(prob_support_labels_reshaped, [None, T, dim])

        x, att_scores = main_attention_layer(query=attended, key=key,
                                             value=prob_support_labels_reshaped, return_attention_scores=True)

        x1 = tf.keras.layers.Lambda(lambda x: x[:, 0, :], name='sensor_1_attended_attended')(x)
        x2 = tf.keras.layers.Lambda(lambda x: x[:, 1, :], name='sensor_2_attended_attended')(x)
        x3 = tf.keras.layers.Lambda(lambda x: x[:, 2, :], name='sensor_3_attended_attended')(x)
        mean = tf.reduce_mean(x, axis=1)
        mean = tf.keras.layers.Softmax()(mean)
        out_1, out_2, out_3 = x1, x2, x3
        # Apply argmax to the mean tensor
        x = tf.expand_dims(tf.argmax(mean, axis=-1, output_type=tf.int32), axis=-1)
        # Convert the argmax indices to float and scale to max_weight
        x = tf.cast(x, tf.float32) * (max_weight / (tf.cast(tf.shape(mean)[-1], tf.float32) - 1))

        # Reshape x to match the expected output shape
        # x = tf.expand_dims(x, axis=-1)
        out = x

    else:
        final_dense_layer = tf.keras.layers.Dense(1, activation=final_activation, name='final_dense')
        out_1 = max_weight * final_dense_layer(x1)
        out_2 = max_weight * final_dense_layer(x2)
        out_3 = max_weight * final_dense_layer(x3)
        if use_sensor_attention:
            out = max_weight * final_dense_layer(mean)
        else:

            out = tf.reduce_mean(
                tf.concat([tf.expand_dims(out_1, axis=2), tf.expand_dims(out_2, axis=2), tf.expand_dims(out_3, axis=2)],
                          axis=2), axis=2)
            out = tf.keras.layers.Flatten()(out)

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = tf.keras.Model(inputs=inputs,
                           outputs=[out, out_1, out_2, out_3]
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
        if use_sensor_attention:
            if use_weighted_loss:
                model.compile(loss=[None, WeightedMeanSquaredError(weight_loss_multipliers_dict),
                                    WeightedMeanSquaredError(weight_loss_multipliers_dict),
                                    WeightedMeanSquaredError(weight_loss_multipliers_dict)],
                              metrics=['mae', 'mse'],
                              optimizer=opt)
            if use_probabilistic_app:
                model.compile(loss=[None, CustomSquaredLoss(max_weight, smpl_rate),
                                    CustomSquaredLoss(max_weight, smpl_rate),
                                    CustomSquaredLoss(max_weight, smpl_rate)],
                              metrics=['mae', 'mse'],
                              optimizer=opt)
            else:
                model.compile(loss=[None, tf.keras.losses.MeanSquaredError(),
                                    tf.keras.losses.MeanSquaredError(),
                                    tf.keras.losses.MeanSquaredError()],
                              metrics=['mae', 'mse'],
                              optimizer=opt)
        else:
            loss_fn = FlexibleCrossEntropy(max_weight, 80)
            model.compile(optimizer=opt, loss=[None, loss_fn, loss_fn, loss_fn], metrics=['accuracy'])

    return model
