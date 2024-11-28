import os
import pandas as pd
import tensorflow as tf
from itertools import combinations
import keras
import keras.ops as K
def get_weight_files(directory):
    weight_data = []

    # Walk through directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if file matches criteria
            if 'weight' in file.lower() and file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    # Read the CSV file
                    df = pd.read_csv(file_path)
                    # Check if 'snc_1' column exists

                    snc1 = df['Snc1'].values
                    snc2 = df['Snc2'].values
                    snc3 = df['Snc3'].values
                    weight_data.append({
                        'file': file,
                        'snc_1': snc1, 'snc_2': snc2, 'snc_3': snc3,
                    })
                except Exception as e:
                    print(f"Error reading {file}: {e}")

    return weight_data

def frame_all_snc(snc_data, frame_length=100, frame_step=18):
    framed_snc_data=[]
    for dict in snc_data:
        framed_snc1 =  tf.signal.frame(dict['snc_1'],
                        frame_length,  # Number of samples in each frame
                        frame_step,    # Number of samples to step between frames (for overlap)
                        pad_end=False)
        framed_snc2 = tf.signal.frame(dict['snc_2'],
                                      frame_length,  # Number of samples in each frame
                                      frame_step,  # Number of samples to step between frames (for overlap)
                                      pad_end=False)
        framed_snc3 = tf.signal.frame(dict['snc_3'],
                                      frame_length,  # Number of samples in each frame
                                      frame_step,  # Number of samples to step between frames (for overlap)
                                      pad_end=False)
        framed_snc_data.append({
                        'file': dict['file'],
                        'framed_snc1': framed_snc1, 'framed_snc2': framed_snc2, 'framed_snc3': framed_snc3,
                    })

    return framed_snc_data


def batch_pearson_correlation(x, y):
    # x, y shape: (batch_size, 100)

    # Calculate means for each sample
    x_mean = tf.reduce_mean(x, axis=1, keepdims=True)  # shape: (batch_size, 1)
    y_mean = tf.reduce_mean(y, axis=1, keepdims=True)  # shape: (batch_size, 1)

    # Center the variables
    x_centered = x - x_mean  # shape: (batch_size, 100)
    y_centered = y - y_mean  # shape: (batch_size, 100)

    # Calculate numerator (covariance)
    numerator = tf.reduce_sum(x_centered * y_centered, axis=1)  # shape: (batch_size,)

    # Calculate denominators (standard deviations)
    x_std = tf.sqrt(tf.reduce_sum(tf.square(x_centered), axis=1))  # shape: (batch_size,)
    y_std = tf.sqrt(tf.reduce_sum(tf.square(y_centered), axis=1))  # shape: (batch_size,)

    # Calculate correlations
    correlations = (numerator / (x_std * y_std)).numpy()  # shape: (batch_size,)

    return correlations
def sensor_pearson_correlation(framed_signal_dict):
    corr_data = []
    global_min_corr = 1
    worst_dict = {}
    for dict in framed_signal_dict:
        corr= {}
        for i, j in combinations(range(3), 2):
            corr[f'{i+1}_{j+1}'] = batch_pearson_correlation(dict[f'framed_snc{i+1}'],
                                                             dict[f'framed_snc{i+1}'])
        min_corr = K.min(K.min([corr['1_2'], corr['1_3'], corr['2_3']], axis=-1)).numpy()
        if min_corr < global_min_corr:
            global_min_corr = min_corr
        worst_dict = dict
        corr_data.append({
            'file': dict['file'],
            'corr': corr,
            'min_corr': min_corr
        })
    return corr_data, global_min_corr, worst_dict


if __name__ == "__main__":
    file_dir = r"C:\Users\sofia.a\PycharmProjects\DATA_2024\Sorted (1)\Sorted"
    snc_data = get_weight_files(file_dir)

    frame_len = 50
    framed_snc_data = frame_all_snc(snc_data, frame_length=frame_len, frame_step=18)
    corr_data, global_min_corr, worst_dict = sensor_pearson_correlation(framed_snc_data)

    tt = 1
