from models import create_attention_weight_distr_estimation_model
from db_generators.generators import create_data_for_model
from constants import *

from utils.get_data import get_weight_file_from_dir
from keras.callbacks import TensorBoard
from custom.callbacks import OutputPlotCallback
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
    attention_distr_snc_model_parameters_dict = {'window_size_snc': 512,
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
                                                 'loss_balance':0.5,
                                           'sensor_fusion': 'attention',
                                           }

    model = create_attention_weight_distr_estimation_model(**attention_distr_snc_model_parameters_dict)

    model.summary()
    print(model.output_names)


    file_dir = '/home/wld-algo-6/DataCollection/Data'
    person_dict = get_weight_file_from_dir(file_dir)
    labels_to_balance = [0, 0.5, 1, 2]
    epoch_len = 5
    persons_for_train_initial_model = ['Leeor']#['Avihoo', 'Aviner', 'Shai']
    train_ds = create_data_for_model(person_dict, 512, 1024, labels_to_balance, epoch_len,
                                     used_persons=persons_for_train_initial_model, data_mode='Train')
    val_ds = train_ds

    out_callback = OutputPlotCallback(person_dict, log_dir,
                                      samples_per_label_per_person=10, used_persons=['Leeor'], picture_name='Leeor',
                                      data_mode='Test',
                                      phase='test')
    history = model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        callbacks=[out_callback,
            TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard')),
        ],
        epochs=30,
        validation_data=val_ds,
        verbose=1,
    )

    print(history.history.keys())

    window_size = model.input['snc_1'].shape[-1]
    # max_weight = 2
    # smpl_rate = 9
    # fixed_sigma = 0.001
    #
    # y_pred = tf.convert_to_tensor([[1, 0.3], [0.4, 1]])
    # y_true = tf.convert_to_tensor([0.5, 1])
    # # loss = GaussianCrossEntropyLoss(smpl_rate=smpl_rate, max_weight=max_weight, fixed_sigma=0.001)(y_true, y_pred)
    #
    # # Ensure inputs are float32
    # y_true = tf.cast(y_true, tf.float32)
    # y_pred = tf.cast(y_pred, tf.float32)
    #
    # # Extract predicted mu and sigma
    # pred_mu = y_pred[:, 0]
    # pred_sigma = y_pred[:, 1]
    #
    # # tf.print('y_true.shape',y_true.shape)
    # # tf.print('y_pred.shape', y_pred.shape)
    #
    # # Create true and predicted distributions
    # true_distribution = create_distribution(y_true, tf.ones_like(y_true) * fixed_sigma, smpl_rate=smpl_rate)
    # true_distribution = tf.nn.softmax(true_distribution, axis=-1)
    # # true_distribution = tf.one_hot(tf.cast(y_true, tf.int32), self.smpl_rate)
    # pred_distribution = create_distribution(pred_mu, pred_sigma,  smpl_rate=smpl_rate)
    #
    # cross_entropy = keras.losses.CategoricalCrossentropy()(true_distribution, pred_distribution)
    ttt = 1

    find_max_sigma(p=0.5, max_weight=2)