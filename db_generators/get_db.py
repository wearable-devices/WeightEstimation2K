import tensorflow as tf
from constants import *
from db_generators.generators import MultiInputGenerator
import pandas as pd
import os
import re
def extract_fn_Snc(example):
    features = {
        'Snc1': tf.io.FixedLenFeature(SNC_WINDOW_SIZE, tf.float32),
        'Snc2': tf.io.FixedLenFeature(SNC_WINDOW_SIZE, tf.float32),
        'Snc3': tf.io.FixedLenFeature(SNC_WINDOW_SIZE, tf.float32),
        'SncButton': tf.io.FixedLenFeature(int(SNC_WINDOW_SIZE / SNC_ADC_PACKET), tf.float32),
    }

    parsed_features = tf.io.parse_example(example, features)
    features = {'Snc1': parsed_features['Snc1'], 'Snc2': parsed_features['Snc2'], 'Snc3': parsed_features['Snc3'],
                }

    # Create labels directly in one-hot format for binary classification
    labels = parsed_features['SncButton']
    return features, labels

def process_file(file_path):
    df = pd.read_csv(file_path)  # Read the file into a DataFrame
    snc1_data = df.loc[df['Snc1'].notnull(), 'Snc1'].values#[54:54+378]#[254*18:]#[:378] #df['Snc1'].values  # Extract the 'snc1' column values as a numpy array
    snc2_data = df.loc[df['Snc2'].notnull(), 'Snc2'].values#[54:54+378]#[254*18:]#[:378] # Extract the 'snc2' column values as a numpy array
    snc3_data = df.loc[df['Snc3'].notnull(), 'Snc3'].values#[54:54+378]#[254*18:]#[:378]  # Extract the 'snc3' column values as a numpy array


    return snc1_data, snc2_data, snc3_data








if __name__ == "__main__":
    file_dir = '/home/wld-algo-5/Data/17_7_2024 data base'
    persons_dict = get_weight_file_from_dir(file_dir)
    # pdict = take_db(persons_dict)
    labels_to_balance = [0,1,2,4,6,8]
    train_ds = MultiInputGenerator(persons_dict, window_size=306, batch_size=1024, labels_to_balance=labels_to_balance)
    train_ds.__getitem__(0)
    ttt = 1