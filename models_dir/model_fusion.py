import keras
import keras.ops as K
from pandas.conftest import axis_frame
from custom.layers import MajorityVote

from models import get_optimizer

def one_sensor_model_majorityvote_fusion(indent, pattern_len, snc_model_1, snc_model_2, snc_model_3,
                                         fusion_type='average',
                                     window_size_snc=234,
                                    trainable=False,
                                     optimizer='Adam', learning_rate=0.0016,
                                     ):
    '''fusion_type could be 'average', 'majority_vote' '''
    snc_model_1._name = 'snc_model_1'
    snc_model_2._name = 'snc_model_2'
    snc_model_3._name = 'snc_model_3'

    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=window_size_snc, name='Snc1')
    input_layer_snc2 = keras.Input(shape=window_size_snc, name='Snc2')
    input_layer_snc3 = keras.Input(shape=window_size_snc, name='Snc3')

    snc_model_1.trainable = trainable
    snc_model_2.trainable = trainable
    snc_model_3.trainable = trainable

    snc_output_1 = snc_model_1([input_layer_snc1, input_layer_snc2, input_layer_snc3])[0]
    snc_output_2 = snc_model_2([input_layer_snc1, input_layer_snc2, input_layer_snc3])[0]
    snc_output_3 = snc_model_3([input_layer_snc1, input_layer_snc2, input_layer_snc3])[0]

    if fusion_type == 'average':
        fused_out = K.mean(K.concat([snc_output_1, snc_output_2, snc_output_3]), axis=-1)
    elif fusion_type == 'majority_vote':
        fused_out = MajorityVote()([snc_output_1, snc_output_2, snc_output_3])
    # majority_vote = MajorityVoteLayer(name='majority_vote')(snc_output_1, snc_output_2, snc_output_3)
    #confidence= GlobalConfidenceLayer(pattern_len=pattern_len)(snc_output_1)

    # total_out = K.concat([K.expand_dims(snc_output_1,axis=1),K.expand_dims(snc_output_2, axis=1),
    #                        K.expand_dims(snc_output_3, axis=1)], axis=1)

    model = keras.Model(inputs=[input_layer_snc1, input_layer_snc2, input_layer_snc3],
                                outputs=[fused_out, snc_output_1, snc_output_2, snc_output_3
                                         ],
                           name='snc_fusion_model')

    opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate)
    if trainable:
        model.compile(
            optimizer=opt,
            loss=['mse', 'mse', 'mse', 'mse'
                 ],
            metrics=['mae','mae','mae','mae'],

        )


    return model