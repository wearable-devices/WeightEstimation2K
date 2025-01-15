import tensorflow as tf
from fontTools.ttLib.tables.S_T_A_T_ import table_S_T_A_T_
from oauthlib.uri_validate import query
from scipy.constants import value

from custom.layers import *
# import keras.ops as K
import keras
from custom.losses import *
from utils.special_functions import  find_max_sigma
from custom.metrics import SigmaMetric

def get_optimizer(optimizer = 'LAMB',learning_rate=0.001, weight_decay=0.0):
    # https://stackoverflow.com/questions/67286051/how-can-i-tune-the-optimization-function-with-keras-tuner
    # if optimizer == 'LAMB':
    #     opt = tfa.optimizers.LAMB(learning_rate=learning_rate, beta_1=0.9, weight_decay=weight_decay)
    if optimizer == 'Adam':
        opt = tf.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
    elif optimizer == 'SGD':
        opt = tf.optimizers.SGD(learning_rate=learning_rate)
    else:
        opt = optimizer

    return opt

def get_loss(loss='mse'):
    if loss == 'mse':
        loss = keras.losses.MeanSquaredError()
    elif loss == 'Huber':
        loss = keras.losses.Huber()
    return loss

def get_metric(metric='mae'):
    if metric == 'mae':
        metric = keras.metrics.MeanAbsoluteError()

    return metric


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
            time_attention_layer_1 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
                                                                      name=f'time_attention1_for_sensor_{sensor_num}')
            time_attention_layer_2 = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim,
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
                                             scattering_max_order=1,
                                             units=10, dense_activation='tanh', use_attention=True,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             # use_sensor_attention=False,

                                             sensor_fusion='early',
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
                                             weight_decay=0.0, max_weight=0.3, compile=True,
                                             use_weighted_loss=True, normalization_factor=3,
                                             weight_loss_multipliers_dict={weight: 1 for weight in [0, 1, 2, 4, 6, 8]}
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    scattered_snc1, scattered_snc11 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc1)
    scattered_snc2, scattered_snc22 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc2)
    scattered_snc3, scattered_snc33 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc3)


    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))
    else:
        S_snc1 = K.squeeze(scattered_snc1, axis=-1)
        S_snc2 = K.squeeze(scattered_snc2, axis=-1)
        S_snc3 = K.squeeze(scattered_snc3, axis=-1)

    S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
    S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
    S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))

    # Apply Time attention
    sensor_1 = sensor_attention_processing(S_snc1, units=units, conv_activation=dense_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering, apply_tfp=apply_tfp,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=1, use_attention=use_attention, )
    sensor_2 = sensor_attention_processing(S_snc2, units=units, conv_activation=dense_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering, apply_tfp=apply_tfp,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=2, use_attention=use_attention)
    sensor_3 = sensor_attention_processing(S_snc3, units=units, conv_activation=dense_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering, apply_tfp=apply_tfp,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=3, use_attention=use_attention)

    x1 = sensor_1
    x2 = sensor_2
    x3 = sensor_3

    sensor_conc = K.concatenate([K.expand_dims(x1, axis=1), K.expand_dims(x2, axis=1), K.expand_dims(x3, axis=1)],
                                axis=1)

    if sensor_fusion == 'early':
        fused_x = keras.layers.Flatten()(sensor_conc)
        x = keras.layers.Dense(units, activation=dense_activation, name=f'my_dense_1')(fused_x)
        x = keras.layers.Dropout(0.1)(x)
        x1 = x
        x2 = x
        x3 = x
        mean = x
    elif sensor_fusion == 'attention':
        if use_sensor_ordering:
            sensor_attention_layer = OrderedAttention(num_heads=num_sensor_attention_heads,
                                                      key_dim=key_dim_for_sensor_att, scale_activation=scale_activation)
        else:
            sensor_attention_layer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=key_dim_for_sensor_att)

        attended, _ = sensor_attention_layer(sensor_conc, sensor_conc, return_attention_scores=True)
        x1 = keras.layers.Lambda(lambda x: x[:, 0, :], name='sensor_1_attended')(attended)
        x2 = keras.layers.Lambda(lambda x: x[:, 1, :], name='sensor_2_attended')(attended)
        x3 = keras.layers.Lambda(lambda x: x[:, 2, :], name='sensor_3_attended')(attended)
        mean = K.mean(attended, axis=1)
    else: # sensor_fusion == 'mean'
        mean = K.mean(sensor_conc, axis=1)

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

        x1 = keras.layers.Lambda(lambda x: x[:, 0, :], name='sensor_1_attended_attended')(x)
        x2 = keras.layers.Lambda(lambda x: x[:, 1, :], name='sensor_2_attended_attended')(x)
        x3 = keras.layers.Lambda(lambda x: x[:, 2, :], name='sensor_3_attended_attended')(x)
        mean = K.mean(x, axis=1)
        mean = keras.layers.Softmax()(mean)
        out_1, out_2, out_3 = x1, x2, x3
        # Apply argmax to the mean tensor
        x = tf.expand_dims(tf.argmax(mean, axis=-1, output_type=tf.int32), axis=-1)
        # Convert the argmax indices to float and scale to max_weight
        x = tf.cast(x, tf.float32) * (max_weight / (tf.cast(tf.shape(mean)[-1], tf.float32) - 1))

        # Reshape x to match the expected output shape
        # x = tf.expand_dims(x, axis=-1)
        out = x

    else:
        final_dense_layer = keras.layers.Dense(1, activation=final_activation, name='final_dense')
        out_1 = max_weight * final_dense_layer(x1)
        out_2 = max_weight * final_dense_layer(x2)
        out_3 = max_weight * final_dense_layer(x3)

        out = max_weight * final_dense_layer(mean)

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=[out, out_1, out_2, out_3]
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

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
            model.compile(loss=[None, keras.losses.MeanSquaredError(),
                                keras.losses.MeanSquaredError(),
                                keras.losses.MeanSquaredError()],
                          metrics=[
                              ['mae', 'mse'],  # metrics for first output
                              None,  # no metrics for second output
                              None,  # no metrics for third output
                              None  # no metrics for fourth output
                          ],
                          optimizer=opt)
        # else:
        #     loss_fn = FlexibleCrossEntropy(max_weight, 80)
        #     model.compile(optimizer=opt, loss=[None, loss_fn, loss_fn, loss_fn], metrics=['accuracy'])

    return model

# class MuMAE(keras.metrics.Metric):
#     def __init__(self, name='mu_mae', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.mae = self.add_weight(name='mae', initializer='zeros')
#         self.count = self.add_weight(name='count', initializer='zeros')
#
#     def update_state(self, y_true, y_pred, sample_weight=None):
#         mu = y_pred[:, 0]  # Get mu from combined output
#         error = K.abs(y_true - mu)
#         self.mae.assign_add(K.mean(error))
#         self.count.assign_add(1)
#
#     def result(self):
#         return self.mae / self.count

def create_attention_weight_distr_estimation_model(window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             scattering_max_order=1,

                                             scattering_type='old',
                                             units=10, dense_activation='tanh', use_attention=True,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             # use_sensor_attention=False,
                                             smpl_rate = 10,

                                             sensor_fusion='attention',
                                             use_sensor_ordering=False,
                                             final_activation='sigmoid',
                                             num_heads=4, key_dim_for_snc=4, key_dim_for_sensor_att=4,
                                             scale_activation='linear',
                                             num_sensor_attention_heads=2,

                                             max_sigma = 0.6,

                                             apply_noise=False, stddev=0.1,
                                             optimizer='LAMB', learning_rate=0.0016,
                                             loss_balance = 0.5,
                                             weight_decay=0.0, max_weight=2,
                                                   loss_normalize=False,
                                                   compile=True,
                                             ):
    '''sensor_fusion could be 'early, attention or mean

    scattering_type coukd be 'old' or 'SEMG' '''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')
    if scattering_type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif scattering_type == 'SEMG':
        scattering_layer = SEMGScatteringTransform()

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)


    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))
    else:
        if scattering_type == 'old':
            scattered_snc1 = K.squeeze(scattered_snc1, axis=-1)
            scattered_snc2 = K.squeeze(scattered_snc2, axis=-1)
            scattered_snc3 = K.squeeze(scattered_snc3, axis=-1)

    S_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
    S_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
    S_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))

    # Apply Time attention
    sensor_1 = sensor_attention_processing(S_snc1, units=units, conv_activation=dense_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=1, use_attention=use_attention, )
    sensor_2 = sensor_attention_processing(S_snc2, units=units, conv_activation=dense_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=2, use_attention=use_attention)
    sensor_3 = sensor_attention_processing(S_snc3, units=units, conv_activation=dense_activation,
                                           attention_layers_for_one_sensor=attention_layers_for_one_sensor,
                                           num_heads=num_heads, key_dim=key_dim_for_snc,
                                           use_time_ordering=use_time_ordering,
                                           apply_noise=apply_noise,
                                           stddev=stddev,
                                           sensor_num=3, use_attention=use_attention)

    x1 = sensor_1
    x2 = sensor_2
    x3 = sensor_3

    sensor_conc = K.concatenate([K.expand_dims(x1, axis=1), K.expand_dims(x2, axis=1), K.expand_dims(x3, axis=1)],
                                axis=1)

    max_sigma =  max_sigma#tf.cast(find_max_sigma(p=0.5, max_weight = max_weight), dtype=tf.float32)

    if sensor_fusion == 'early':
        fused_x = keras.layers.Flatten()(sensor_conc)
        x = keras.layers.Dense(units, activation=dense_activation, name=f'my_dense_1')(fused_x)
        x = keras.layers.Dropout(0.1)(x)
        x1 = x
        x2 = x
        x3 = x
        mean = x
    elif sensor_fusion == 'attention':
        if use_sensor_ordering:
            sensor_attention_layer = OrderedAttention(num_heads=num_sensor_attention_heads,
                                                      key_dim=key_dim_for_sensor_att, scale_activation=scale_activation)
        else:
            sensor_attention_layer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=key_dim_for_sensor_att)

        attended, sensor_score_matrix = sensor_attention_layer(sensor_conc, sensor_conc, return_attention_scores=True)
        x1 = keras.layers.Lambda(lambda x: x[:, 0, :], name='sensor_1_attended')(attended)
        x2 = keras.layers.Lambda(lambda x: x[:, 1, :], name='sensor_2_attended')(attended)
        x3 = keras.layers.Lambda(lambda x: x[:, 2, :], name='sensor_3_attended')(attended)
        mean = K.mean(attended, axis=1)
        variance_evaluation_layer = keras.layers.Conv2D(filters=3,kernel_size=(3,3), strides=(1, 1),)
        score_matrix = K.transpose(sensor_score_matrix, axes=(0,2,3,1))
        coeff_for_sigma = variance_evaluation_layer(score_matrix)
        coeff_for_sigma = keras.layers.Flatten()(coeff_for_sigma)
        coeff_for_sigma = keras.layers.Softmax(axis=-1)(coeff_for_sigma)


    else: # sensor_fusion == 'mean'
        mean = K.mean(sensor_conc, axis=1)


    final_dense_layer = keras.layers.Dense(1, activation=final_activation, name='final_dense')
    variance_layer_1 = keras.layers.Dense(1, activation='sigmoid', name='variance_layer_1')
    # variance_layer_2 = keras.layers.Dense(1, activation='sigmoid', name='variance_layer_2')

    out = max_weight * final_dense_layer(mean)

    conf_1 =max_sigma* variance_layer_1(x1)
    conf_2 = max_sigma* variance_layer_1(x2)
    conf_3 = max_sigma* variance_layer_1(x3)
    conc_sigma = K.concatenate([conf_1, conf_2, conf_3], axis = -1)
    # sigma = max_sigma * variance_layer_2(K.concatenate([conf_1, conf_2,conf_3], axis = 1))
    sigma =keras.layers.Dot(axes=1)([coeff_for_sigma, conc_sigma])

    # Create a single output that combines both
    combined_output = keras.layers.Concatenate(name='gaussian_output')([out, sigma])

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    # model = keras.Model(inputs=inputs,
    #                     outputs={'mean_output': out,
    #                              'gaussian_output': combined_output}
    #                     )

    model = keras.Model(inputs=inputs,
                        outputs={
                            'weight_output': keras.layers.Lambda(lambda x: x, name='weight_output')(out),  # For mean output
                            # keras.layers.Lambda(lambda x: x, name='gaussian_output')(combined_output)
                            'gaussian_output':combined_output
                            # For gaussian output
                        })
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

        losses = {
            'weight_output': 'mse',  # or you can use None if you don't want to train on it
            'gaussian_output': GaussianCrossEntropyLoss(smpl_rate=smpl_rate, max_weight=max_weight, fixed_sigma=0.001, normalize=loss_normalize)#GaussianNLLLoss()
        }

        # If you want to ignore mean_output during training, use loss_weights
        loss_weights = {
            'weight_output':  loss_balance,  # Set weight to 0 to ignore during training
            'gaussian_output': 1-loss_balance
        }

        model.compile(
            loss=losses,
            loss_weights=loss_weights,  # This ensures only gaussian_output affects training
            metrics={
                'gaussian_output': [SigmaMetric()],
                'weight_output': [
                    keras.metrics.MeanAbsoluteError(name='mae'),
                    keras.metrics.MeanSquaredError(name='mse')
                ]
            },
            optimizer=opt
        )

    return model


def create_early_fusion_weight_estimation_model(window_size_snc=306, apply_tfp=False,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             scattering_max_order=1,
                                             units=10, dense_activation='tanh',
                                             final_activation='sigmoid',
                                             optimizer='LAMB', learning_rate=0.0016,
                                             weight_decay=0.0, max_weight=0.3, compile=True,
                                             use_weighted_loss=True, normalization_factor=3,
                                             weight_loss_multipliers_dict={weight: 1 for weight in [0, 1, 2, 4, 6, 8]}
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    scattered_snc1, scattered_snc11 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc1)
    scattered_snc2, scattered_snc22 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc2)
    scattered_snc3, scattered_snc33 = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)(
        input_layer_snc3)


    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))
    else:
        S_snc1 = K.squeeze(scattered_snc1, axis=-1)
        S_snc2 = K.squeeze(scattered_snc2, axis=-1)
        S_snc3 = K.squeeze(scattered_snc3, axis=-1)

    S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
    S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
    S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))

    #eraly fusion
    fused = K.concatenate([S_snc1,S_snc2,S_snc3], axis=2)

    # Apply Time attention
    mean =  K.mean(fused, axis=1)
    x = keras.layers.Dense(units, activation=dense_activation)(mean)
    x = keras.layers.Dense(units, activation=dense_activation)(x)


    final_dense_layer = keras.layers.Dense(1, activation=final_activation, name='final_dense')


    out = max_weight * final_dense_layer(mean)

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=out
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)


        model.compile(loss=keras.losses.MeanSquaredError(),
                      metrics=
                          ['mae', 'mse'],  #
                      optimizer=opt)
        # else:
        #     loss_fn = FlexibleCrossEntropy(max_weight, 80)
        #     model.compile(optimizer=opt, loss=[None, loss_fn, loss_fn, loss_fn], metrics=['accuracy'])

    return model

def mean_time_sensor_image(sensor_1_image):

    x_mean = tf.reduce_mean(sensor_1_image, axis=1)# K.mean(sensor_1_image, axis=1)



    return x_mean

def create_average_sensors_weight_estimation_model(window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             scattering_max_order=1,
                                             units=10, dense_activation='tanh', use_attention=True,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             # use_sensor_attention=False,
                                            scattering_type='old',

                                             final_activation='sigmoid',

                                             # apply_noise=True, stddev=0.1,
                                             optimizer='LAMB', learning_rate=0.0016,
                                             weight_decay=0.0, max_weight=0.3, compile=True,
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    if scattering_type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif scattering_type == 'SEMG':
        scattering_layer = SEMGScatteringTransform()

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)

    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))
    else:
        if scattering_type == 'old':
            scattered_snc1 = K.squeeze(scattered_snc1, axis=-1)
            scattered_snc2 = K.squeeze(scattered_snc2, axis=-1)
            scattered_snc3 = K.squeeze(scattered_snc3, axis=-1)

    S_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
    S_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
    S_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))

    # Apply Time attention

    x1 = mean_time_sensor_image(S_snc1)
    x2 = mean_time_sensor_image(S_snc2)
    x3 = mean_time_sensor_image(S_snc3)

    x1 = keras.layers.Dense(units, activation=dense_activation, name='dense_1')(x1)
    x2 = keras.layers.Dense(units, activation=dense_activation, name='dense_2')(x2)
    x3 = keras.layers.Dense(units, activation=dense_activation, name='dense_3')(x3)
    # final_dense_layer = keras.layers.Dense(1, activation=final_activation, name='final_dense')
    out_1 = max_weight * keras.layers.Dense(1, activation=final_activation, name='final_dense_1')(x1)
    out_2 = max_weight *  keras.layers.Dense(1, activation=final_activation, name='final_dense_2')(x2)
    out_3 = max_weight *  keras.layers.Dense(1, activation=final_activation, name='final_dense_3')(x3)

    # out = AverageTwoClosest()([ K.squeeze(out_1, axis=-1) ,K.squeeze(out_2, axis=-1), K.squeeze(out_3, axis=-1)])
    out = 1/3*(K.squeeze(out_1, axis=-1)+ K.squeeze(out_2, axis=-1)+K.squeeze(out_3, axis=-1))

    # Add dimension back to make shapes consistent
    # out = K.expand_dims(out, axis=-1)

    out_1 = K.squeeze(out_1, axis=-1)
    out_2 = K.squeeze(out_2, axis=-1)
    out_3 = K.squeeze(out_3, axis=-1)

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=[out, out_1, out_2, out_3]
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)


        model.compile(loss=[None, 'mse',
                            'mse',
                            'mse'],
            # loss=[None, keras.losses.MeanSquaredError(),
            #                 keras.losses.MeanSquaredError(),
            #                 keras.losses.MeanSquaredError()],
                      metrics=[
                          ['mae', 'mse'],  # metrics for first output
                          'mse',  # no metrics for second output
                          None,  # no metrics for third output
                          None  # no metrics for fourth output
                      ],
                      optimizer=opt)
        # else:
        #     loss_fn = FlexibleCrossEntropy(max_weight, 80)
        #     model.compile(optimizer=opt, loss=[None, loss_fn, loss_fn, loss_fn], metrics=['accuracy'])

    return model

def create_one_sensors_weight_estimation_model(sensor_num=2, window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             scattering_max_order=1,
                                             units=10, dense_activation='relu', use_attention=True,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             # use_sensor_attention=False,
                                            scattering_type='old',

                                             final_activation='sigmoid',

                                             # apply_noise=True, stddev=0.1,
                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.0, max_weight=3.0, compile=True,
                                               loss = 'mse',
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    if scattering_type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif scattering_type == 'SEMG':
        scattering_layer = SEMGScatteringTransform()

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)

    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))
    else:
        if scattering_type == 'old':
            scattered_snc1 = K.squeeze(scattered_snc1, axis=-1)
            scattered_snc2 = K.squeeze(scattered_snc2, axis=-1)
            scattered_snc3 = K.squeeze(scattered_snc3, axis=-1)

    S_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
    S_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
    S_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))

    all_sensors = [S_snc1,S_snc2,S_snc3]
    x = all_sensors[sensor_num-1]

    # Apply Time attention

    x = mean_time_sensor_image(x)
    # x_mean = K.mean(x, axis=1)
    # x_min = K.min(x, axis=1)
    # x_max = K.max(x, axis=1)
    # x = K.concatenate([K.expand_dims(x_min, axis=2), K.expand_dims(x_mean, axis=2),
    #                            K.expand_dims(x_max, axis=2)], axis=2)
    x = keras.layers.Dense(units, activation=dense_activation)(x)
    # x = keras.layers.Flatten()(x)


    x = keras.layers.Dense(units, activation=dense_activation, name='dense_1')(x)



    out = (max_weight) * keras.layers.Dense(1, activation=final_activation, name='final_dense_1')(x)


    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=out
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
        model_loss = get_loss(loss)

        model.compile(loss= model_loss,
                      metrics=
                          ['mae', 'mse'],
                      optimizer=opt,
                      run_eagerly=True)


    return model


class DummyLoss(keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        return 0.0 * keras.backend.mean(y_pred)



def one_sensors_weight_estimation_proto_model(sensor_num=2, window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             scattering_max_order=1,
                                             units=10, dense_activation='relu', use_attention=True,
                                             key_dim_for_time_attention=5,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             # use_sensor_attention=False,
                                            scattering_type='old',
                                             final_activation='sigmoid',
                                             add_noise=True,

                                             # apply_noise=True, stddev=0.1,
                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.0, max_weight=3.0, compile=True,
                                               loss = 'mse',
                                             loss_balance = 0.5
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    if scattering_type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif scattering_type == 'SEMG':
        scattering_layer = SEMGScatteringTransform(undersampling=undersampling)

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)

    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))
    else:
        if scattering_type == 'old':
            scattered_snc1 = K.squeeze(scattered_snc1, axis=-1)
            scattered_snc2 = K.squeeze(scattered_snc2, axis=-1)
            scattered_snc3 = K.squeeze(scattered_snc3, axis=-1)

            S_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
            S_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
            S_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))
        else :
            S_snc1 = scattered_snc1
            S_snc2 = scattered_snc2
            S_snc3 = scattered_snc3

    all_sensors = [S_snc1,S_snc2,S_snc3]
    if sensor_num == 'all':
        x = keras.layers.Concatenate(axis=2, name='sensor_oncatenate')(all_sensors)
    else:
        x = all_sensors[sensor_num-1]

    # Apply Time attention
    if use_attention:
        for _ in range(attention_layers_for_one_sensor):
            x = keras.layers.MultiHeadAttention(num_heads=3,key_dim=key_dim_for_time_attention)(query=x,key = x,value = x)
    x = mean_time_sensor_image(x)


    if add_noise:
        x = GaussianNoiseLayer(stddev=0.1)(x)
    x = keras.layers.Dense(units, activation=dense_activation, name='dense_1')(x)
    x = keras.layers.Dense(units, activation=dense_activation, name='dense_2')(x)
    # x = keras.layers.Flatten()(x)


    # x = keras.layers.Dense(units, activation=dense_activation, name='dense_1')(x)
    # x = keras.layers.Dense(units, activation=dense_activation)(x)



    # out = [(max_weight) * keras.layers.Dense(1, activation=final_activation, name='final_dense_1')(x), x]
    out_0 = (max_weight) * keras.layers.Dense(1, activation=final_activation, name='final_dense_1')(x)
    # out_1 = keras.layers.Dense(units, name='embedding_output')(x)  # Give embedding a specific shape

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                        outputs=out_0
                        )

    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
        model_loss = get_loss(loss)

        # losses = {
        #     'out_weight': model_loss,
        #     'out_embd': DummyLoss()  # instead of None
        # }

        # loss_weights = {
        #     'out_weight': loss_balance,
        #     'out_embd': 1 - loss_balance
        # }

        # metrics = {
        #     'out_weight': ['mae', 'mse'],
        #     'out_embd': None
        # }

        model.compile(loss=model_loss,
                      # loss_weights=loss_weights,
                      metrics=['mae', 'mse'],
                      optimizer=opt,
                      run_eagerly=True)




    return model



def one_sensor_weight_estimation_with_zeroidhint_model(sensor_num=2, window_size_snc=306,
                                                       hint_model='',
                                             undersampling=3,
                                             scattering_max_order=1,
                                             units=10, dense_activation='relu', use_attention=True,
                                             attention_layers_for_one_sensor=1,
                                             use_time_ordering=False,
                                             final_activation='sigmoid',

                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.0, max_weight=2, compile=True,
                                               loss = 'Huber',
                                             loss_balance = 0.5
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain}


    user_id = hint_model([input_layer_snc1, input_layer_snc2, input_layer_snc3]) # (batch_size, embd_dim)

    scattering_layer = SEMGScatteringTransform(undersampling=undersampling)

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)

    if scattering_max_order == 2:
        scaterred_snc_list = [K.squeeze(scattered_snc1, axis=-1),
                              K.squeeze(scattered_snc2, axis=-1),
                              K.squeeze(scattered_snc3, axis=-1)]
        scattered_snc_list_2 = [K.squeeze(scattered_snc11, axis=-1),
                                K.squeeze(scattered_snc22, axis=-1),
                                K.squeeze(scattered_snc33, axis=-1)]

        S_snc1 = K.concatenate((scaterred_snc_list[0], scattered_snc_list_2[0]), axis=1)
        # S_snc1 = K.transpose(S_snc1, axes=(0, 2, 1))
        S_snc2 = K.concatenate((scaterred_snc_list[1], scattered_snc_list_2[1]), axis=1)
        # S_snc2 = K.transpose(S_snc2, axes=(0, 2, 1))
        S_snc3 = K.concatenate((scaterred_snc_list[2], scattered_snc_list_2[2]), axis=1)
        # S_snc3 = K.transpose(S_snc3, axes=(0, 2, 1))


    S_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
    S_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
    S_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))

    all_sensors = [S_snc1,S_snc2,S_snc3]
    x = all_sensors[sensor_num-1]

    # Apply Time attention

    x = mean_time_sensor_image(x)

    # embd_dim = user_id.shape[1]
    x = keras.layers.Dense(units, activation=dense_activation, name='dense_1')(x)
    y = keras.layers.Dense(units,activation=dense_activation, name='dense_2')(user_id)

    # x = x-user_id
    # mult = tf.matmul(x,y)
    conc = x+y #K.concatenate([x,y], axis=-1)

    out = [(max_weight) * keras.layers.Dense(1, activation=final_activation, name='final_dense_1')(conc), x]
    # out = [(max_weight) * mult, x]

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=out
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
        model_loss = get_loss(loss)

        model.compile(loss= [model_loss,ProtoLoss(number_of_persons=4, proto_meaning='weight')],
                      loss_weights=[ loss_balance, 1 - loss_balance],
                      metrics=[['mae', 'mse'], None],
                      optimizer=opt,
                      run_eagerly=True)


    return model