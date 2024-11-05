
from keras.callbacks import TensorBoard
import optuna
import os

from optuna_dashboard import save_plotly_graph_object, save_note
from datetime import datetime
from optuna.trial import TrialState
from models import *
from pathlib import Path
from db_generators.generators import MultiInputGenerator, convert_generator_to_dataset, create_data_for_model
from utils.get_data import get_weight_file_from_dir
from constants import *
from custom.callbacks import *
# from db_generators.create_person_dict import *
from tensorflow.keras.callbacks import ModelCheckpoint
from custom.callbacks import get_layer_output

file_dir = '/home/wld-algo-6/DataCollection/Data'
persons_dict = get_weight_file_from_dir(file_dir)

snc_window_size = 162
batch_size = 1024
labels_to_balance = [0, 500,1000,2000]
epoch_len = 5
data_mode='Test'
used_persons='all'
ex = MultiInputGenerator(
        persons_dict,
        window_size=snc_window_size,
        batch_size=batch_size,
        data_mode=data_mode,
        labels_to_balance=labels_to_balance,
        epoch_len=epoch_len,
        person_names=used_persons)

ex.__getitem__(0)
ex.__len__()



model_snc_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/05-11-2024-15-16-39/trials/trial_0/initial_pre_trained_model.keras'
custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain}
model = tf.keras.models.load_model(model_snc_path, custom_objects=custom_objects,
                                               compile=True,
                                               safe_mode=False)

#
# used_persons = 'all'
# if used_persons == 'all':
#     persons_dict = persons_dict
# else:
#     persons_dict = filter_dict_by_keys(persons_dict, used_persons)
window_size = model.input['snc_1'].shape[-1]

# First, print all layer names to find the ones you need
for layer in model.layers:
    print(layer.name)




samples_per_label_per_person=10


output_dict = {person: {} for person in persons_dict.keys()}
for person, weight_dict in persons_dict.items():

    for weight, records in weight_dict.items():
        if data_mode == 'all':
            used_records = records
        else:
            used_records = [record for record in records if record['phase'] == data_mode]
        if len(used_records)>0:
            snc1_batch = []
            snc2_batch = []
            snc3_batch = []
            for _ in range(samples_per_label_per_person):
                # Randomly select a file for this label
                file_idx = tf.random.uniform([], 0, len(used_records), dtype=tf.int32)
                file_data = used_records[file_idx.numpy()]
                # Generate a random starting point within the file
                start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - window_size + 1,
                                          dtype=tf.int32)
                # Extract the window
                snc1_batch.append(file_data['snc_1'][start:start + window_size])
                snc2_batch.append(file_data['snc_2'][start:start + window_size])
                snc3_batch.append(file_data['snc_3'][start:start + window_size])
            # if len(self.model.inputs == 3): # model doesn't have labeled input
            persons_input_data = [tf.stack(snc1_batch), tf.stack(snc2_batch), tf.stack(snc3_batch)]
            layer_out = get_layer_output(model, persons_input_data, 'dense')
            predictions = model([persons_input_data])[0]
            # predictions = tf.squeeze(tf.keras.layers.Flatten()(predictions), axis=-1)
            output_dict[person][weight] = predictions


ttt = 1

import tensorflow as tf
print("TensorFlow version:", tf.__version__)
