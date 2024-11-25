
from keras.callbacks import TensorBoard
import optuna
import os

from optuna_dashboard import save_plotly_graph_object, save_note
from datetime import datetime
from optuna.trial import TrialState
from models import *
from pathlib import Path
from db_generators.generators import create_data_for_userId_model,UserIdModelGenerator
from utils.get_data import get_weight_file_from_dir
from constants import *
from custom.callbacks import *
# from db_generators.create_person_dict import *
from tensorflow.keras.callbacks import ModelCheckpoint
from custom.callbacks import get_layer_output
from custom.for_debug import WeightMonitorCallback

from db_generators.generators import preprocess_person_data
def ex_model(sensor_num=2, window_size_snc=306, dense_activation='relu',
                                            embd_dim=5,number_of_persons=10,
                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.01,  compile=True,
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    x = input_layer_snc2
    # x = keras.layers.BatchNormalization()(x)
    out = keras.layers.Dense(embd_dim, activation=dense_activation,kernel_initializer='he_normal', name = 'person_id_final')(x)

    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=out
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

        model.compile(loss=ProtoLoss(number_of_persons=number_of_persons,
                                     temperature=1),  #'categorical_crossentropy',
                      optimizer=opt,
                      run_eagerly=True)


    return model
def logging_dirs():
    package_directory = Path(__file__).parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir


snc_window_size = 306

logs_root_dir, log_dir = logging_dirs()
file_dir = r"C:\Users\sofia.a\PycharmProjects\DATA_2024\Sorted"#'/home/wld-algo-6/DataCollection/Data'
person_dict = get_weight_file_from_dir(file_dir)
person_zero_dict = {person_name: weight_dict[0] for person_name, weight_dict in person_dict.items() if 0 in weight_dict.keys()}

persons = ['Alisa', 'Asher2', 'Avihoo', 'Aviner', 'HishamCleaned','Lee',
               'Leeor',
               #'Daniel',
               'Liav',
                'Foad',
               'Molham',
               'Ofek'
               ]#,'Guy']
train_data, test_data, person_to_idx = preprocess_person_data(person_zero_dict,
                                                                  window_size_snc=window_size_snc,
                                                                  window_step=54,
                                                                  persons=persons,
                                                                  print_stat=False,
                                                                  multiplier=10)

# After preprocessing
# if check_for_nans(train_data) or check_for_nans(test_data):
#     raise ValueError("NaN or Inf values found in the preprocessed data")

# Convert labels to one-hot encoding
num_persons = len(person_to_idx) if persons == 'all' else len(persons)


# Create and train model
model = ex_model(
    sensor_num=2,
    number_of_persons=num_persons,
    window_size_snc=snc_window_size, embd_dim=10,
    dense_activation='tanh',
    learning_rate=0.0016,
    # Match your input size
)
model.summary()

# Create person to index mapping
person_to_idx = {name: idx for idx, name in enumerate(persons)}
gen = UserIdModelGenerator(person_zero_dict, person_to_idx, window_size=window_size_snc,
                           data_mode='Train', batch_size=1024,
                             person_names=persons, epoch_len=None,
                             contacts = ['M'])
gen.__getitem__(0)

epoch_len = None
batch_size = 110
train_ds = create_data_for_userId_model(person_zero_dict, person_to_idx, snc_window_size, batch_size,
                                 epoch_len, persons,
                                 data_mode='Train', contacts=['M'])
val_ds = create_data_for_userId_model(person_zero_dict,person_to_idx, snc_window_size, batch_size,
                                       epoch_len,persons,
                                      data_mode='Test',contacts=['M'])
model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[NanCallback(), WeightMonitorCallback(
                            print_freq='epoch',  # 'epoch' or 'batch'
                            layer_wise=True,     # Print stats for each layer
                            detailed_stats=True  # Print detailed statistics
)],
        epochs=1000,
        batch_size=128
    )
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
