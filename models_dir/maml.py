import tensorflow as tf
import keras
from keras import layers
import numpy as np
from custom.layers import *
import keras.ops as K
from models import get_optimizer, get_loss, get_metric
from models_dir.model_fusion import one_sensor_model_fusion

class MAML:
    def __init__(
            self,
            model_fn,
            loss = 'Huber',
            metrics=['mae'],
            inner_learning_rate=0.01,
            outer_learning_rate=0.001,
            inner_batch_size=1,
            inner_steps=1,
            model_parameters_dict={}
            # model_parameters_dict = {}
    ):
        """
        Initialize MAML with the model architecture and hyperparameters

        Args:
            model_fn: Function that returns a Keras model
            inner_learning_rate: Learning rate for task-specific adaptation
            outer_learning_rate: Learning rate for meta-update
            inner_batch_size: Batch size for inner loop training
            inner_steps: Number of gradient steps for adaptation
        """
        self.model = model_fn(**model_parameters_dict)#(**model_parameters_dict)
        self.inner_learning_rate = inner_learning_rate
        self.outer_learning_rate = outer_learning_rate
        self.inner_batch_size = inner_batch_size
        self.inner_steps = inner_steps
        self.loss = get_loss(loss)
        self.metrics = {metric: get_metric(metric) for metric in metrics}#['mae']

        # Initialize meta-optimizer
        self.meta_optimizer = keras.optimizers.Adam(learning_rate=outer_learning_rate)

    def inner_loop(self, support_data, support_labels):
        """
        Perform inner loop adaptation with multiple steps
        """
        task_parameters = [tf.identity(w) for w in self.model.trainable_variables]
        current_parameters = task_parameters

        # Multiple adaptation steps
        for _ in range(self.inner_steps):
            with tf.GradientTape() as tape:
                # Temporarily set current parameters
                for w, w_current in zip(self.model.trainable_variables, current_parameters):
                    w.assign(w_current)

                predictions = self.model(support_data)[0]
                loss = self.loss(support_labels, predictions)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, self.model.trainable_variables)
            current_parameters = [
                w - self.inner_learning_rate * g
                for w, g in zip(current_parameters, gradients)
            ]

        return current_parameters, loss

    def outer_loop(self, tasks_batch):
        """
        Perform outer loop update across multiple tasks

        Args:
            tasks_batch: List of (support_data, support_labels, query_data, query_labels)
        """
        with tf.GradientTape() as tape:
            total_metrics  = []
            total_loss = 0
            total_metrics = {metric_name:0 for metric_name in self.metrics.keys()}
            for support_snc1, support_snc2, support_snc3, support_labels, query_snc1, query_snc2, query_snc3, query_labels in tasks_batch:
                # Inner loop adaptation
                support_data = [support_snc1, support_snc2, support_snc3]
                updated_parameters, _ = self.inner_loop(support_data, support_labels)

                # Temporarily update model parameters
                original_parameters = [tf.identity(w) for w in self.model.trainable_variables]
                for w, w_updated in zip(self.model.trainable_variables, updated_parameters):
                    w.assign(w_updated)

                # Compute loss on query set
                query_data = [query_snc1, query_snc2, query_snc3]
                predictions = self.model(query_data)[0]
                query_loss = self.loss(
                    query_labels, predictions
                )
                query_metrics = {}
                for metric_name, metric_fn in self.metrics.items():
                   query_metrics[metric_name] = metric_fn(query_labels, predictions)
                   total_metrics[metric_name] += query_metrics[metric_name]
                # query_metrics = [metric(query_labels, predictions) for metric in self.metrics]
                # for i, metric in enumerate(self.metrics):
                #     total_metrics[i]+=tf.reduce_mean(query_metrics[i])
                total_loss += tf.reduce_mean(query_loss)


                # Restore original parameters
                for w, w_original in zip(self.model.trainable_variables, original_parameters):
                    w.assign(w_original)

            mean_loss = total_loss / len(tasks_batch)
            mean_metrics = {metric_name: metric_value/len(tasks_batch) for metric_name, metric_value in total_metrics.items()}

        # Compute meta-gradients
        meta_gradients = tape.gradient(mean_loss, self.model.trainable_variables)

        # Apply meta-update
        self.meta_optimizer.apply_gradients(
            zip(meta_gradients, self.model.trainable_variables)
        )

        return mean_loss, {name :m.numpy() for name,m in mean_metrics.items()}

    def adapt_to_task(self, support_data, support_labels):
        """
        Adapt the model to a new task using the support set

        Args:
            support_data: Support set data
            support_labels: Support set labels

        Returns:
            Updated parameters for the new task
        """
        return self.inner_loop(support_data, support_labels)


# Example usage
def create_one_sensor_model(window_size_snc = 256, scattering_type='SEMG',
                      J_snc=5, Q_snc=(2, 1),
                      undersampling=4.8,
                      use_attention=True,
                      attention_layers_for_one_sensor=2, key_dim_for_time_attention=5,
                      units=10, dense_activation='relu',
                      max_weight=2, final_activation='sigmoid',
                      sensor_num=2,
                      optimizer='Adam',
                      learning_rate=0.0016,weight_decay=0.0,
                      loss = 'Huber'):
    """Create a simple neural network model"""
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    if scattering_type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif scattering_type == 'SEMG':
        scattering_layer = SEMGScatteringTransform(undersampling=undersampling)

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)

    if scattering_type == 'old':
        scattered_snc1 = K.squeeze(scattered_snc1, axis=-1)
        scattered_snc2 = K.squeeze(scattered_snc2, axis=-1)
        scattered_snc3 = K.squeeze(scattered_snc3, axis=-1)

        S_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
        S_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
        S_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))
    else:
        S_snc1 = scattered_snc1
        S_snc2 = scattered_snc2
        S_snc3 = scattered_snc3

    all_sensors = [S_snc1, S_snc2, S_snc3]
    x = all_sensors[sensor_num - 1]
    if use_attention:
        for _ in range(attention_layers_for_one_sensor):
            x = keras.layers.MultiHeadAttention(num_heads=3,key_dim=key_dim_for_time_attention)(query=x,key = x,value = x)
    x = K.mean(x, axis=1)

    x = keras.layers.Dense(units, activation=dense_activation)(x)
    x = keras.layers.Dense(units, activation=dense_activation)(x)
    out = [(max_weight) * keras.layers.Dense(1, activation=final_activation, name='final_dense_1')(x)]#, x]

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                        outputs=out
                        )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
        model_loss = get_loss(loss)

        model.compile(loss=[model_loss],#,
                            #ProtoLoss(number_of_persons=4, proto_meaning='weight')],
                      # loss_weights=[loss_balance, 1 - loss_balance],
                      metrics=
                      ['mae', 'mse'],
                      optimizer=opt,
                      run_eagerly=True)
    return model

def create_maml_model(window_size_snc = 256, scattering_type='SEMG',
                      J_snc=5, Q_snc=(2, 1),
                      undersampling=4.8,
                      use_attention=True,
                      attention_layers_for_one_sensor=2, key_dim_for_time_attention=5,
                      units=10, dense_activation='relu',
                      max_weight=2, final_activation='sigmoid',
                      optimizer='Adam',
                      learning_rate=0.0016,weight_decay=0.0,
                      loss = 'Huber'):
    one_sensor_parameters = {'window_size_snc' : window_size_snc,
                             'scattering_type':scattering_type,
                      'J_snc':J_snc, 'Q_snc':Q_snc,
                      'undersampling':undersampling,
                      'use_attention':use_attention,
                      'attention_layers_for_one_sensor':attention_layers_for_one_sensor, 'key_dim_for_time_attention':key_dim_for_time_attention,
                      'units':units, 'dense_activation':dense_activation,
                      'max_weight':max_weight, 'final_activation':final_activation,
                      'optimizer':optimizer,
                      'learning_rate':learning_rate,'weight_decay':weight_decay,
                      'loss': loss}
    snc_model_1 = create_one_sensor_model(sensor_num=1,**one_sensor_parameters)
    snc_model_2 = create_one_sensor_model(sensor_num=2,**one_sensor_parameters)
    snc_model_3 = create_one_sensor_model(sensor_num=3, **one_sensor_parameters)
    model = one_sensor_model_fusion(snc_model_1, snc_model_2, snc_model_3,
                            fusion_type='majority_vote',
                            window_size_snc=window_size_snc,
                            use_sensor_ordering=True, num_sensor_attention_heads=2,
                            max_weight=max_weight,
                            trainable=True,
                            optimizer=optimizer, learning_rate=learning_rate,
                            loss=loss,
                            compile=False
                            )
    return model
