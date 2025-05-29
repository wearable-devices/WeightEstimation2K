import numpy as np
import matplotlib.pyplot as plt
from pyriemann.channelselection import ElectrodeSelection
from pyriemann.estimation import Covariances
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from pyriemann.classification import MDM
from mne.datasets import sample
from mne import read_epochs_eeglab


# Let's create some example EEG data
# In a real scenario, you'd load your own EEG data
# Shape: (n_trials, n_channels, n_times)
def create_synthetic_eeg_data(n_trials=100, n_channels=64, n_times=1000):
    # Create random EEG-like data
    data = np.random.randn(n_trials, n_channels, n_times)

    # Create binary labels (0, 1)
    labels = np.random.randint(0, 2, n_trials)

    # Simulate more discriminative patterns in certain channels for class 1
    # Channels 5, 10, 15, 20, 25 are more important for discrimination
    important_channels = [1,2]
    for i in range(n_trials):
        if labels[i] == 1:
            for ch in important_channels:
                # Add a distinctive pattern
                data[i, ch, :] += 0.5 * np.sin(np.linspace(0, 10 * np.pi, n_times))

    return data, labels



import tensorflow as tf

def are_tensors_equal(tensor1, tensor2):
    # Check if shapes are equal first
    if tensor1.shape != tensor2.shape:
        return False

    # Check if all values are equal
    # reduce_all() ensures ALL values are True
    return tf.reduce_all(tf.equal(tensor1, tensor2))

def get_user_sensors(user_name):
    gen = CSVInputGenerator(loaded_dict, window_size,
                            transition_balance={'00': 32,
                                                '01': 3,
                                                '10': 3,
                                                '11': 26},  # data_mode='all',
                            batch_size=1024,
                            person_names=[user_name], epoch_len=None)

    # take user data
    # loaded_dict[user_name]['press_release']
    batch = gen.__getitem__(0)
    x_1 = tf.concat([tf.expand_dims(batch[0]['snc_1'], axis=1), tf.expand_dims(batch[0]['snc_2'], axis=1),
                     tf.expand_dims(batch[0]['snc_3'], axis=1)], axis=1)
    x_1 = tf.cast(x_1, dtype=tf.float32).numpy()
    y_1 = batch[1]
    X, y = create_synthetic_eeg_data(n_trials=1125, n_channels=n_channels, n_times=306)

    # Compute covariance matrices
    cov = Covariances().fit_transform(X)

    # Initialize electrode selection with 16 electrodes and Riemannian metric
    electrode_selector = ElectrodeSelection(nelec=n_channels, metric='riemann', n_jobs=1)

    # Fit the electrode selector to the data
    electrode_selector.fit(cov, y)

    # Get the selected channels
    selected_channels = electrode_selector.subelec_

    # Transform the data to keep only the selected channels
    X_selected = electrode_selector.transform(cov)
    return selected_channels, X_selected

# from spd_statistic.files import *

from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann
import pandas as pd

def get_framed_snc_data_from_file(file_path, window_size=64, frame_step=16, apply_zpk2sos_filter=False):
    data = pd.read_csv(file_path)

    # Extract sensors
    snc1 = data['Snc1'].values
    snc2 = data['Snc2'].values
    snc3 = data['Snc3'].values


    if apply_zpk2sos_filter:
        snc1 = zpk2sos_filter(snc1)
        snc2 = zpk2sos_filter(snc2)
        snc3 = zpk2sos_filter(snc3)


    signal = tf.concat([tf.expand_dims(tf.convert_to_tensor(snc1), axis=-1),
                        tf.expand_dims(tf.convert_to_tensor(snc2), axis=-1),
                        tf.expand_dims(tf.convert_to_tensor(snc3), axis=-1)], axis=-1)

    framed_signal = tf.signal.frame(signal, window_size,
                                    frame_step=frame_step,
                                    pad_end=False,
                                    pad_value=0,
                                    axis=-2,
                                    name=None
                                    )
    framed_signal = tf.transpose(framed_signal,perm=(0,2,1))

    return framed_signal, snc1, snc2, snc3

if __name__ == "__main__":
    # Generate synthetic data
    n_channels = 3
    window_size = 306
    pattern_len = 306

    file_Alisa_4 = '/home/wld-algo-6/Data/SortedCleaned/Alisa/press_release/Alisa_4_press_0_Nominal_TableTop_M.csv'

    # train_files = [file_Alisa_2]
    test_files = [file_Alisa_4]

    # get labeled data from csv filles
    # train_data, train_labels = get_framed_labeled_date_from_files(test_files, window_size=window_size,
    #                                                               pattern_len=pattern_len,
    #                                                               apply_zpk2sos_filter=True)

    framed_signal, snc1, snc2, snc3 = get_framed_snc_data_from_file(file_Alisa_4, window_size=window_size, frame_step=16, apply_zpk2sos_filter=False)
    # train_data = framed_signal.numpy()

    csd_layer = SequentialCrossSpectralDensityLayer_pyriemann(
        # for_model=True,
        return_sequence=False,
        main_window_size=window_size,  # window size
        # preprocessing_type=None,  # or 'fft'
        take_tangent_proj=True,
        frame_step=8,
        frame_length=90,

        base_point_calculation='identity',
    )

    csd_layer.call(framed_signal)

    ttt=1

