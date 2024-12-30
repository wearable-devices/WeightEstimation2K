import tensorflow as tf
import keras
from person_models import create_person_zeroid_model
from models_dir.model_fusion import one_sensor_model_fusion

class EmbeddingNetwork(keras.Model):
    def __init__(self, window_size_snc=306,
                     J_snc=5, Q_snc=(2, 1),
                     undersampling=2,#4.8
                     units=10, dense_activation='relu',
                    scattering_type='SEMG',#'old',
                    embd_dim=5,
                    number_of_persons=10,
                     optimizer='Adam', learning_rate=0.0016,
                     weight_decay=0.01):
        super().__init__()
        self.model_parameters_dict = {'window_size_snc':window_size_snc,
                                             'J_snc': J_snc, 'Q_snc': Q_snc,'undersampling':undersampling,'units':units, 'dense_activation':dense_activation,
                                            'scattering_type': scattering_type,'embd_dim':embd_dim,'number_of_persons':number_of_persons,
                                             'optimizer':optimizer, 'learning_rate':learning_rate,'weight_decay':weight_decay}
        self.encoder_sensor_1 = create_person_zeroid_model(sensor_num=1,compile=True, **self.model_parameters_dict )
        self.encoder_sensor_2 = create_person_zeroid_model(sensor_num=2, compile=True, **self.model_parameters_dict)
        self.encoder_sensor_3 = create_person_zeroid_model(sensor_num=3, compile=True, **self.model_parameters_dict)

        self.fused_encoder = one_sensor_model_fusion(self.encoder_sensor_1, self.encoder_sensor_2, self.encoder_sensor_3,
                             fusion_type='concatenate',
                             window_size_snc=window_size_snc,
                             trainable=False,
                            compile=False)

        # User embedding layer
        self.user_embedding = keras.layers.Embedding(
            input_dim=20,#num_users,  # Total number of users
            output_dim=3*embd_dim#embedding_dim
        )

    def call(self, x, user_id):
        # x = [snc1,snc2,snc3]
        # Get feature embedding
        feature_embedding = self.fused_encoder(x)[0]

        # Get user embedding
        user_emb = self.user_embedding(user_id)

        # Combine feature and user embeddings
        # You can use concatenation, addition, or more sophisticated fusion
        combined = feature_embedding + user_emb
        return combined


# class PrototypeLayer(keras.layers.Layer):
#     def __init__(self, num_prototypes):
#         super().__init__()
#         self.num_prototypes = num_prototypes
#
#     def build(self, input_shape):
#         self.prototypes = self.add_weight(
#             shape=(self.num_prototypes, input_shape[-1]),
#             initializer='glorot_uniform',
#             trainable=True,
#             name='prototypes'
#         )
#
#     def call(self, embeddings):
#         # Calculate distances to prototypes
#         distances = tf.reduce_sum(
#             tf.square(
#                 tf.expand_dims(embeddings, 1) - tf.expand_dims(self.prototypes, 0)
#             ),
#             axis=-1
#         )
#         return -distances


class WeightEstimationWithUserEmbeddings(keras.Model):
    def __init__(self, max_weight=2,  window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=2,#4.8
                                             units=10, dense_activation='relu',
                                            scattering_type='SEMG',
                                            embd_dim=5,

                                            number_of_persons=10,
                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.01):
        super().__init__()
        # Define input layers
        self.snc1_input = keras.layers.Input(shape=(window_size_snc,), name='snc_1')
        self.snc2_input = keras.layers.Input(shape=(window_size_snc,), name='snc_2')
        self.snc3_input = keras.layers.Input(shape=(window_size_snc,), name='snc_3')
        self.user_input = keras.layers.Input(shape=(), dtype=tf.int32, name='users')

        self.embedding_network = EmbeddingNetwork( window_size_snc=window_size_snc,
                                             J_snc=J_snc, Q_snc=Q_snc,
                                             undersampling=undersampling,#4.8
                                             units=units, dense_activation=dense_activation,
                                            scattering_type=scattering_type,
                                            embd_dim=embd_dim,
                                            number_of_persons=number_of_persons,
                                             optimizer=optimizer, learning_rate=learning_rate,
                                             weight_decay=weight_decay)
        self.decision_layer = keras.layers.Dense(units=1, activation='sigmoid')
        self.max_weight = max_weight
        # self.prototype_layer = PrototypeLayer(num_prototypes)

    def call(self, inputs):
        # Unpack inputs dictionary
        snc1 = inputs['snc_1']
        snc2 = inputs['snc_2']
        snc3 = inputs['snc_3']
        user_ids = inputs['users']
        # Get embeddings that combine feature and user information
        embeddings = self.embedding_network([snc1, snc2, snc3], user_ids)
        # Calculate distances to prototypes and convert to logits
        # logits = self.prototype_layer(embeddings)
        weight_prediction = self.max_weight * self.decision_layer(embeddings)
        return weight_prediction

    def model(self):
        # Create a Keras Model that you can compile and fit
        inputs = {
            'snc_1': self.snc1_input,
            'snc_2': self.snc2_input,
            'snc_3': self.snc3_input,
            'users': self.user_input
        }
        outputs = self.call(inputs)
        return keras.Model(inputs=inputs, outputs=outputs)


def build_model_for_new_user(model, user, embedding_dim=15, optimizer='Adam', learning_rate = 0.0016):
    '''
    Args:
        model:
        user:
        embedding_dim:
        optimizer:
        learning_rate:

    Returns:

    '''
    # Initialize user embedding
    user_embedding = tf.Variable(tf.random.normal([1, embedding_dim]))


#
# def adapt_to_new_user(model, support_set, query_set, embedding_dim=15, learning_rate=0.01):
#     # Initialize user embedding
#     user_embedding = tf.Variable(tf.random.normal([1, embedding_dim]))
#
#     for epoch in range(adaptation_steps):
#         with tf.GradientTape() as tape:
#             # Compute prototypes from support set
#             support_embeddings = model.embedding_network(
#                 support_set['x'], user_embedding)
#             prototypes = compute_prototypes(support_embeddings, support_set['y'])
#
#             # Compute loss on query set
#             query_embeddings = model.embedding_network(
#                 query_set['x'], user_embedding)
#             logits = compute_distances(query_embeddings, prototypes)
#             loss = tf.keras.losses.sparse_categorical_crossentropy(
#                 query_set['y'], logits, from_logits=True)
#
#         # Update user embedding
#         grads = tape.gradient(loss, user_embedding)
#         tf.keras.optimizers.Adam(learning_rate).apply_gradients(
#             [(grads, user_embedding)])
#
#     return user_embedding