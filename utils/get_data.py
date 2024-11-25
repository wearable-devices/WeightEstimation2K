import tensorflow as tf
import pandas as pd
import re
import os
from custom.layers import ScatteringTimeDomain, SEMGScatteringTransform
import tensorflow.keras as tf_keras
#import tensorflow_probability as tfp

# def process_csv(file_path):
#     # Read the CSV file
#     df = pd.read_csv(file_path)
#
#     # Get the column and remove None values
#     values = df['24-bit Value (from Hex)'].dropna().values
#
#     # Use NumPy array slicing to split the values
#     # This takes every 3rd element starting from index 0, 1, and 2 respectively
#     snc1 = values[0::3]  # Start at index 0, step by 3
#     snc2 = values[1::3]  # Start at index 1, step by 3
#     snc3 = values[2::3]  # Start at index 2, step by 3
#
#     return snc1, snc2, snc3

def process_normal_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Get the column and remove None values
    snc_1 = df['Snc1'].dropna().values
    snc_2 = df['Snc2'].dropna().values
    snc_3 = df['Snc3'].dropna().values

    return snc_1, snc_2, snc_3

def get_windowed_data_from_file(file_path, window_size_snc):
    snc1_data, snc2_data, snc3_data = process_normal_csv(file_path)
    # add batch dimention --- axis=0
    snc1_data = tf.expand_dims(snc1_data, axis=0)
    snc2_data = tf.expand_dims(snc2_data, axis=0)
    snc3_data = tf.expand_dims(snc3_data, axis=0)

    snc1_batch_framed = tf.signal.frame(snc1_data, frame_length=window_size_snc, frame_step=18, axis=-1)
    snc2_batch_framed = tf.signal.frame(snc2_data, frame_length=window_size_snc, frame_step=18, axis=-1)
    snc3_batch_framed = tf.signal.frame(snc3_data, frame_length=window_size_snc, frame_step=18, axis=-1)

    # Assuming snc1 and snc2 have shape (batch_size, time_steps, features)
    batch_size, time_steps, _ = snc1_batch_framed.shape

    # Reshape inputs to (batch_size * time_steps, features)
    snc1_reshaped = tf.reshape(snc1_batch_framed, [-1, snc1_batch_framed.shape[-1]])
    snc2_reshaped = tf.reshape(snc2_batch_framed, [-1, snc2_batch_framed.shape[-1]])
    snc3_reshaped = tf.reshape(snc3_batch_framed, [-1, snc3_batch_framed.shape[-1]])

    return snc1_reshaped, snc2_reshaped, snc3_reshaped

def get_weight_file_from_dir(file_dir):
    """
       Recursively search and process weight measurement CSV files from a directory structure.

       This function searches for CSV files that match the pattern 'Leeor_[weight]kg/g_horizontsal.csv'
       and processes them into a structured dictionary. Weights in grams are automatically converted
       to kilograms.

       Args:
           file_dir (str): Root directory path to search for CSV files

       Returns:
           dict: Nested dictionary structure:
               {
                   'person_name': {
                       weight_in_kg: [
                           {
                               'name': str,       # Formatted file name
                               'snc_1': Any,      # Processed data from CSV
                               'snc_2': Any,      # Processed data from CSV
                               'snc_3': Any,      # Processed data from CSV
                               'file_path': str   # Full path to file
                           },
                           ...
                       ],
                       ...
                   },
                   ...
               }

       Notes:
           - Files must follow naming convention: 'Leeor_XXkg_horizontsal.csv' or 'Leeor_XXg_horizontsal.csv'
           - Requires process_csv() function to parse CSV contents
           - Progress messages are printed during processing
           - All weights are stored in kilograms in the output dictionary
       """
    persons_dict = {}
    for dirpath, dirnames, filenames in os.walk(file_dir):
        # phase = dirpath.split('/')[-1]
        phase = dirpath.split('\\')[-1]
        for filename in filenames:
            if re.search(r'weight', filename) and filename.endswith('.csv'):
                # username_numexp_expname_label_force_orientation_contact
                try:
                    person, numexp, expname, label, force, orientation, contact = filename.split('_')
                except:
                    print(filename)
                    continue
                contact = contact[0]
                label = float(label)/1000
                # person = dirpath.split('/')[-1]
                file_path = os.path.join(dirpath, filename)
                file_name = '_'.join(file_path.split('/')[-3:])

                # Extract the number from the filename
                # match_kg = re.search(r'Leeor_(\d+)kg_horizontsal', filename)
                # match_g = re.search(r'weight_(\d+)g_horizontsal', filename)
                # if match_kg or match_g:
                weight = label#float(match_kg.group(1)) if match_kg else float(match_g.group(1))/1000
                snc1_data, snc2_data, snc3_data = process_normal_csv(file_path)

                if person not in persons_dict:
                    persons_dict[person] = {}
                if weight not in persons_dict[person]:
                    persons_dict[person][weight] = []

                persons_dict[person][weight].append({
                    'name': file_name,
                    'snc_1': snc1_data,
                    'snc_2': snc2_data,
                    'snc_3': snc3_data,
                    'file_path': file_path,
                    'contact': contact,
                    'phase': phase
                })

                print(f"Processing file: {file_path}, Weight: {weight}kg")

    return persons_dict


def mean_scattering_snc(persons_dict, window_size=162, samples_per_weight_per_person=5,
                        J_snc=6, Q_snc=(2,1), undersampling=5, contacts=['L', 'M', 'R'], scattering_type='old'):
    '''takes persons_dict, takes samples_per_weight_per_person=5 windows for each snc sensor, applyes scattering and returns obtained dictionary
    scattering_type could be 'old' or 'SEMG'
    '''
    persons_names = [person_key + contact for person_key in persons_dict for contact in contacts]
    output_dict = {person: {} for person in persons_names}
    for person, weight_dict in persons_dict.items():
        # weight_dict_part = {weight: weight_dict[weight] for weight in self.considered_weights if weight in weight_dict}
        # label = weight
        for contact in contacts:
            for weight, records in weight_dict.items():
                used_records = [record for record in records if record['contact'] == contact]
                if len(used_records) > 0:
                    snc1_batch = []
                    snc2_batch = []
                    snc3_batch = []
                    for _ in range(samples_per_weight_per_person):
                        # Randomly select a file for this label
                        file_idx = tf.random.uniform([], 0, len(used_records), dtype=tf.int32)
                        file_data = used_records[file_idx.numpy()]
                        # Generate a random starting point within the file
                        start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - window_size + 1,
                                                  dtype=tf.int32)

                        snc_1_window = file_data['snc_1'][start:start + window_size]
                        snc_2_window = file_data['snc_2'][start:start + window_size]
                        snc_3_window = file_data['snc_3'][start:start + window_size]
                        snc_1_window = tf.cast(snc_1_window, dtype=float) / 2 ** 24
                        snc_2_window = tf.cast(snc_2_window, dtype=float) / 2 ** 24
                        snc_3_window = tf.cast(snc_3_window, dtype=float) / 2 ** 24
                        # Extract the window
                        snc1_batch.append(snc_1_window)
                        snc2_batch.append(snc_2_window)
                        snc3_batch.append(snc_3_window)
                        # labels.append(label)

                    persons_input_data = [tf.stack(snc1_batch), tf.stack(snc2_batch), tf.stack(snc3_batch)]
                    if scattering_type=='old':
                        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling,
                                                                max_order=2)
                    elif scattering_type == 'SEMG':
                        scattering_layer = SEMGScatteringTransform()

                    scattered_snc1, scattered_snc11 = scattering_layer(persons_input_data[0])
                    scattered_snc2, scattered_snc22 = scattering_layer(persons_input_data[1])
                    scattered_snc3, scattered_snc33 = scattering_layer(persons_input_data[2])

                    if scattering_type == 'old':
                        scattered_snc1 = tf.squeeze(scattered_snc1, axis=-1)
                        scattered_snc11 = tf.squeeze(scattered_snc11, axis=-1)

                        scattered_snc2 = tf.squeeze(scattered_snc2, axis=-1)
                        scattered_snc22 = tf.squeeze(scattered_snc22, axis=-1)

                        scattered_snc3 = tf.squeeze(scattered_snc3, axis=-1)
                        scattered_snc33 = tf.squeeze(scattered_snc33, axis=-1)

                    # scattered_snc1_mean = tf.reduce_mean(scattered_snc1, axis=-1)
                    # scattered_snc2_mean = tf.reduce_mean(scattered_snc2, axis=-1)
                    # scattered_snc3_mean = tf.reduce_mean(scattered_snc3, axis=-1)

                    scattered_snc1_mean = tfp.stats.percentile(scattered_snc1, 50.0, axis=-1)
                    scattered_snc2_mean = tfp.stats.percentile(scattered_snc2, 50.0, axis=-1)
                    scattered_snc3_mean = tfp.stats.percentile(scattered_snc3, 50.0, axis=-1)

                    fused_sensors = tf.concat([scattered_snc1_mean, scattered_snc2_mean, scattered_snc3_mean], axis=-1)
                    # fused_sensors = tf.concat([scattered_snc2_mean], axis=-1)
                    output_dict[person+contact][weight] = fused_sensors
    return output_dict




