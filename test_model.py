from models import (create_attention_weight_distr_estimation_model, create_early_fusion_weight_estimation_model,
                    create_average_sensors_weight_estimation_model,create_one_sensors_weight_estimation_model)
from db_generators.generators import create_data_for_model
from constants import *

from utils.get_data import get_weight_file_from_dir
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from custom.callbacks import OutputPlotCallback, SaveKerasModelCallback, FeatureSpacePlotCallback
import os
from pathlib import Path
from datetime import datetime
import tensorflow as tf
from utils.special_functions import find_max_sigma
import keras

from custom.losses import GaussianCrossEntropyLoss

def logging_dirs():
    package_directory = Path(__file__).parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir

def gaussian( x, mu, sigma):
    """Compute Gaussian distribution."""
    x = tf.cast(x, tf.float32)
    mu = tf.cast(mu, tf.float32)
    sigma = tf.cast(sigma, tf.float32)

    return tf.exp(-tf.square(x - mu) / (2 * tf.square(sigma))) / (
                sigma * tf.sqrt(2 * tf.cast(3.14159, tf.float32)))
def create_distribution(mu, sigma, max_weight = 2, smpl_rate=11):
    """Create Gaussian distribution for given mu and sigma."""
    # Create x values
    x_values = tf.linspace(0.0,max_weight, smpl_rate)

    # Reshape tensors for broadcasting
    mu = tf.expand_dims(mu, axis=1)  # (batch_size, 1)
    sigma = tf.expand_dims(sigma, axis=1)  # (batch_size, 1)
    x_values = tf.expand_dims(x_values, axis=0)  # (1, smpl_rate)

    # Calculate distribution
    distribution = gaussian(x_values, mu, sigma)

    # Apply softmax normalization
    return distribution#tf.nn.softmax(distribution, axis=-1)
if __name__ == "__main__":
    logs_root_dir, log_dir= logging_dirs()
    attention_distr_snc_model_parameters_dict = {'window_size_snc': 512, 'scattering_type':'old',
                                           'J_snc': 7,  # trial.suggest_int('J_snc', 5, 7),  # 5,
                                           'Q_snc': (2, 1),
                                           'undersampling': 4.4,
                                           'scattering_max_order': 1,
                                           'use_attention': False,
                                           'attention_layers_for_one_sensor': 1,
                                           'use_sensor_ordering': True,
                                           'units': 6,
                                           'dense_activation': 'linear',
                                                 'smpl_rate':11,
                                           # trial.suggest_categorical('conv_activation', ['linear',  'relu', ]),# trial.suggest_categorical('conv_activation', ['tanh', 'sigmoid', 'relu', 'linear']),#'relu',
                                           'use_time_ordering': True,
                                           'num_heads': 3,
                                           'key_dim_for_snc': 3,
                                           'key_dim_for_sensor_att': 16,
                                           'num_sensor_attention_heads': 1,

                                           'max_sigma': 0.6,
                                           'final_activation': 'sigmoid',
                                           'apply_noise': False,
                                           'max_weight': 2.5,
                                           'optimizer': 'Adam',
                                           'weight_decay': 0,
                                           'learning_rate': 0.0016,
                                                 'loss_balance':1,
                                           'sensor_fusion': 'attention',
                                           }

    average_sensors_weight_estimation_model_dict = {'window_size_snc':512,
                                             'J_snc':7, 'Q_snc': (2, 1),
                                             'undersampling':4.8,
                                             'scattering_max_order':1,
                                             'units':10, 'dense_activation':'linear',
                                                    'use_attention':False,
                                             'attention_layers_for_one_sensor':1,
                                             'use_time_ordering':False,
                                             # use_sensor_attention=False,
                                            'scattering_type':'old',

                                             'final_activation':'tanh',
                                             # 'num_heads':4, 'key_dim_for_snc':4,

                                             # 'apply_noise':True, stddev=0.1,
                                             'optimizer':'Adam', 'learning_rate':0.0016,
                                             'weight_decay':0.0, 'max_weight':2, 'compile':True}

    # model = create_attention_weight_distr_estimation_model(**attention_distr_snc_model_parameters_dict)
    # model = create_early_fusion_weight_estimation_model(optimizer='Adam', window_size_snc=512)
    # model = create_average_sensors_weight_estimation_model(**average_sensors_weight_estimation_model_dict)
    model = create_one_sensors_weight_estimation_model(sensor_num=1, **average_sensors_weight_estimation_model_dict)
    model.summary()
    print(model.output_names)


    file_dir = '/home/wld-algo-6/DataCollection/Data'
    person_dict = get_weight_file_from_dir(file_dir)
    labels_to_balance = [0, 0.5, 1, 2]
    epoch_len = 5
    persons_for_train_initial_model = ['Leeor']#,'Avihoo', 'Aviner', 'Shai']
    train_ds = create_data_for_model(person_dict, 512, 1024, labels_to_balance, epoch_len,
                                     used_persons=persons_for_train_initial_model, data_mode='Train')
    val_ds = create_data_for_model(person_dict, 512, 1024, labels_to_balance, epoch_len,
                                     used_persons=persons_for_train_initial_model, data_mode='Test')

    for person in persons_for_train_initial_model:
        print(f'Working with {person}')
        out_callback = OutputPlotCallback(person_dict, log_dir,
                                      samples_per_label_per_person=10, output_num=0, used_persons=[person], picture_name=person+'0',
                                      data_mode='Test',
                                      phase='test')
        history = model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            callbacks=[out_callback,
                       # OutputPlotCallback(person_dict, log_dir,
                       #                    samples_per_label_per_person=10, output_num=1, used_persons=[person],
                       #                    picture_name=person + '1',
                       #                    data_mode='Test',
                       #                    phase='test'),
                       # OutputPlotCallback(person_dict, log_dir,
                       #                    samples_per_label_per_person=10, output_num=2, used_persons=[person],
                       #                    picture_name=person + '2',
                       #                    data_mode='Test',
                       #                    phase='test'),
                       # OutputPlotCallback(person_dict, log_dir,
                       #                    samples_per_label_per_person=10, output_num=3, used_persons=[person],
                       #                    picture_name=person + '3',
                       #                    data_mode='Test',
                       #                    phase='test'),
                TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard')),
                       # ModelCheckpoint(
                       #     filepath=os.path.join(log_dir,
                       #                           f"pre_trained_model__epoch_{{epoch:03d}}.weights.h5"),
                       #     # Added .weights.h5
                       #     verbose=1,
                       #     save_weights_only=True,
                       #     save_freq='epoch'),
                       SaveKerasModelCallback(log_dir, f"model_epoch{{epoch:03d}}", phase='train'),
                       # FeatureSpacePlotCallback(person_dict, log_dir, layer_name='dense_1', data_mode='Test', proj='pca',
                       #                          metric="euclidean", picture_name_prefix=person + 'test_dict' + f'epoch{{epoch:03d}}',
                       #                          used_persons=[person],
                       #                          num_of_components=3, samples_per_label_per_person=10, phase='test'),
                       #
                       # OutputPlotCallback(person_dict, log_dir,
                       #                    samples_per_label_per_person=10, used_persons=[person], picture_name=person+'train',
                       #                    data_mode='Train',
                       #                    phase='test')

            ],
            epochs=3,
            validation_data=val_ds,
            verbose=1,
        )

        # n_samples = 5

        print(history.history.keys())
        print(history.history)
        # After model.fit
        metrics_values = model.evaluate(
            val_ds,
            # steps=n_samples,  # Use the same number of steps as in training
            # verbose=1
            return_dict=True
        )

        # Since evaluate returns a list of metrics in order of model.metrics_names
        print("Metrics names:", model.metrics_names)
        print("Metrics values:", metrics_values)

        # Or create a dictionary for better readability
        metrics_dict = dict(zip(model.metrics_names, metrics_values))
        print("Loss:", metrics_dict['loss'])  # MSE loss
        print("MAE:", metrics_dict['mae'])  # Mean Absolute Error
        print("MSE:", metrics_dict['mse'])  # Mean Squared Error

    window_size = model.input['snc_1'].shape[-1]



    ttt = 1

    find_max_sigma(p=0.5, max_weight=2)