import keras
import keras.ops as K
# from pandas.conftest import axis_frame
from custom.layers import MajorityVote, OrderedAttention
import tensorflow as tf
from models import get_optimizer, get_loss

def one_sensor_model_fusion(snc_model_1, snc_model_2, snc_model_3,
                             fusion_type='average',
                             window_size_snc=234,
                             use_sensor_ordering=True,num_sensor_attention_heads=2,
                             max_weight=2,
                             trainable=False,
                             optimizer='Adam', learning_rate=0.0016,
                            loss = 'Huber',
                            compile=False
                             ):
    '''fusion_type could be 'average', 'majority_vote', 'attention'  '''

    opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
    loss = get_loss(loss)
    first_loss = loss

    snc_model_1._name = 'snc_model_1'
    snc_model_2._name = 'snc_model_2'
    snc_model_3._name = 'snc_model_3'

    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    snc_model_1.trainable = trainable
    snc_model_2.trainable = trainable
    snc_model_3.trainable = trainable

    snc_output_1 = snc_model_1([input_layer_snc1, input_layer_snc2, input_layer_snc3])[0]
    snc_output_2 = snc_model_2([input_layer_snc1, input_layer_snc2, input_layer_snc3])[0]
    snc_output_3 = snc_model_3([input_layer_snc1, input_layer_snc2, input_layer_snc3])[0]

    fused_out = MajorityVote()([snc_output_1, snc_output_2, snc_output_3])
    if fusion_type == 'average':
        fused_out = K.expand_dims(K.mean(K.concatenate([snc_output_1, snc_output_2, snc_output_3], axis=-1), axis=-1),axis=1)
    elif fusion_type == 'majority_vote':
        fused_out = MajorityVote()([snc_output_1, snc_output_2, snc_output_3])
    elif fusion_type == 'attention':
        layer_for_fusion_1 = snc_model_1([input_layer_snc1, input_layer_snc2, input_layer_snc3])[1]
        layer_for_fusion_2 = snc_model_2([input_layer_snc1, input_layer_snc2, input_layer_snc3])[1]
        layer_for_fusion_3 = snc_model_3([input_layer_snc1, input_layer_snc2, input_layer_snc3])[1]
        key_dim_for_sensor_att = layer_for_fusion_1.shape[-1]
        if use_sensor_ordering:
            sensor_attention_layer = OrderedAttention(num_heads=num_sensor_attention_heads,
                                                      key_dim=key_dim_for_sensor_att, scale_activation='linear')
        else:
            sensor_attention_layer = keras.layers.MultiHeadAttention(num_heads=2, key_dim=key_dim_for_sensor_att)

        sensor_conc = K.concatenate([K.expand_dims(layer_for_fusion_1, axis=1),
                                     K.expand_dims(layer_for_fusion_2, axis=1),
                                     K.expand_dims(layer_for_fusion_3, axis=1)],
                                    axis=1)
        attended, _ = sensor_attention_layer(sensor_conc, sensor_conc, return_attention_scores=True)
        x1 = keras.layers.Lambda(lambda x: x[:, 0, :], name='sensor_1_attended')(attended)
        x2 = keras.layers.Lambda(lambda x: x[:, 1, :], name='sensor_2_attended')(attended)
        x3 = keras.layers.Lambda(lambda x: x[:, 2, :], name='sensor_3_attended')(attended)
        mean = K.mean(attended, axis=1)

        final_dense_layer = keras.layers.Dense(1, activation='sigmoid', name='final_dense')
        snc_output_1 = max_weight * final_dense_layer(x1)
        snc_output_2 = max_weight * final_dense_layer(x2)
        snc_output_3 = max_weight * final_dense_layer(x3)

        fused_out = max_weight * final_dense_layer(mean)
        first_loss = None


    model = keras.Model(inputs=[input_layer_snc1, input_layer_snc2, input_layer_snc3],
                                outputs=[fused_out, snc_output_1, snc_output_2, snc_output_3
                                         ],
                           name='snc_fusion_model')


    if compile:
        model.compile(
            optimizer=opt,
            loss=[first_loss, loss, loss, loss
                 ],
            metrics=['mae','mae','mae','mae'],

        )


    return model
