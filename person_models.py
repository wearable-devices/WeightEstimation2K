import keras
from coremltools.converters.mil.mil.ops.defs.iOS15 import softmax
from keras.src.metrics import CategoricalCrossentropy

from custom.layers import *
import keras.ops as K
from models import get_optimizer, get_loss
from db_generators.generators import preprocess_person_data
from keras.callbacks import TensorBoard
from pathlib import Path
from datetime import datetime
import os

from utils.get_data import get_weight_file_from_dir

def create_person_zeroid_model(sensor_num=2, window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=4.8,
                                             units=10, dense_activation='relu',
                                            scattering_type='old',

                                            number_of_persons=10,
                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.01,  compile=True,
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

    x = K.mean(x, axis=1)
    x = keras.layers.Dense(units, activation=dense_activation, name = 'dense_1')(x)
    x = keras.layers.Dense(units//2, activation=dense_activation, name='person_id')(x)
    out = keras.layers.Dense(number_of_persons, activation='softmax', name = 'person_distr')(x)



    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=out
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

        model.compile(loss= 'categorical_crossentropy',
                      metrics=['accuracy',
                               # keras.metrics.Precision(name='precision'),
                               # keras.metrics.Recall(name='recall'),
                               # keras.metrics.F1Score(name='f1_score')
                               ],
                      optimizer=opt,
                      run_eagerly=True)


    return model
# Create a custom callback to handle multi-value metrics
class CustomTensorBoard(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Convert non-scalar values to scalars if needed
        logs = {k: v.item() if hasattr(v, 'item') else v for k, v in logs.items()}
        super().on_epoch_end(epoch, logs)


def logging_dirs():
    package_directory = Path(__file__).parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir

if __name__ == "__main__":
    logs_root_dir, log_dir = logging_dirs()
    file_dir = '/home/wld-algo-6/DataCollection/Data'
    person_dict = get_weight_file_from_dir(file_dir)
    person_zero_dict = {person_name: weight_dict[0] for person_name, weight_dict in person_dict.items() if 0 in weight_dict.keys()}

    window_size_snc = 512
    # Preprocess data
    train_data, test_data, person_to_idx = preprocess_person_data(person_zero_dict,  window_size_snc=window_size_snc)

    # Convert labels to one-hot encoding
    num_persons = len(person_to_idx)
    train_labels_onehot = keras.utils.to_categorical(train_data['labels'], num_persons)
    test_labels_onehot = keras.utils.to_categorical(test_data['labels'], num_persons)

    # Create and train model
    model = create_person_zeroid_model(
        sensor_num=2,  # Use sensor 2 data
        number_of_persons=num_persons,
        window_size_snc=window_size_snc,
        dense_activation='linear',
        learning_rate=0.016,
        # Match your input size
    )

    callbacks = [
        CustomTensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),

    ]
    # Train the model
    history = model.fit(
        train_data['inputs'],
        train_labels_onehot,
        validation_data=(test_data['inputs'], test_labels_onehot),
        # callbacks=callbacks,
        epochs=200,
        batch_size=128
    )

    ttt=1
    # CategoricalCrossentropy