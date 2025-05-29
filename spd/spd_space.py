import sys
import os

from matplotlib.font_manager import weight_dict

# Add the parent directory of spd_statistic to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from spd_statistic.files import *
# from get_data.get_users_dictionaries import build_windows_dict_from_file
import tensorflow as tf
import numpy as np
from pyriemann.estimation import Covariances
# from spd_statistic.mdm_method import map_binary_strings, extract_channels_tf
# from spd_statistic.spd_pictures import get_lower_trig_part_flattened
# from utility.plotting_functions import interactive_plot_tensors_3d_improved
from pyriemann.tangentspace import TangentSpace
# from utility.plot_from_claude import TensorVisualizer
import plotly.io as pio
# from get_data.get_users_dictionaries import build_user_activity_windows_dict, filter_non_empty_00
from utils.plotting_functions import interactive_plot_tensors_3d_improved

from utils.get_data import get_weight_file_from_dir
import pandas as pd
from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann
from utils.maps import apply_projection_to_dict

def reshape_apply_reshape(input_tensor, cov):
    """
    Reshapes a tensor of shape (n_matrices, n, n_channels, time), applies a function,
    then reshapes the result back appropriately based on the function's output shape.

    Handles two cases:
    1. Function returns (n_matrices, n_channels, n_channels) → result shape: (n_matrices, n, n_channels, n_channels)
    2. Function returns (n_matrices, dim) → result shape: (n_matrices, n, dim)

    Parameters:
    -----------
    input_tensor : tf.Tensor or np.ndarray
        Input tensor with shape (n_matrices, n, n_channels, time)
    function : callable
        Function that takes (n_matrices, n_channels, time) and returns
        either (n_matrices, n_channels, n_channels) or (n_matrices, dim)

    Returns:
    --------
    tf.Tensor or np.ndarray
        Reshaped result with either:
        - Shape (n_matrices, n, n_channels, n_channels) if function returns (n_matrices, n_channels, n_channels)
        - Shape (n_matrices, n, dim) if function returns (n_matrices, dim)
    """
    # Get shapes
    n_matrices, n, n_channels, time = input_tensor.shape

    # Reshape to (n_matrices * n, n_channels, time)
    reshaped_input = tf.reshape(input_tensor, (n_matrices * n, n_channels, time))

    # Apply the function
    result = cov.transform(reshaped_input.numpy())

    # Check the shape of the result to determine how to reshape back
    result_shape = result.shape

    if len(result_shape) == 3:  # Function returned (n_matrices*n, n_channels, n_channels)
        # Reshape back to (n_matrices, n, n_channels, n_channels)
        final_result = tf.reshape(result, (n_matrices, n, n_channels, n_channels))
    elif len(result_shape) == 2:  # Function returned (n_matrices*n, dim)
        # Get the second dimension (dim)
        dim = result_shape[1]
        # Reshape back to (n_matrices, n, dim)
        final_result = tf.reshape(result, (n_matrices, n, dim))
    else:
        raise ValueError(f"Unexpected result shape: {result_shape}. Expected either "
                         f"(n_matrices*n, n_channels, n_channels) or (n_matrices*n, dim)")

    return final_result

def get_spd_matrices_or_proj(data_dict, est='lwf', stat_type='arithmetic',
                             preprocessing_type=None,
                             apply_proj=False,
                             reference_type='identity',  # or 'user_mean', 'total_mean',

                             list_of_channel_indices = [[0, 1 ], [0, 2], [1, 2], [2, 1]],
                             user_names = None, metric='riemann',
                             return_sequences=False,
                             subwindow_size=90, frame_step=8
                             ):
    """if addition_prep==None, we don't calculate freq, so we have only one, if not - we have csd for each freq

    Generate a dictionary of Symmetric Positive Definite (SPD) matrices organized by user, channel
    combinations, and window types.

    This function processes multi-user brain-computer interface (BCI) data to extract covariance or other
    SPD matrices. It can optionally project these matrices to the tangent space, allowing for different
    reference points for projection. The function supports multiple channel combinations and window types.

    Parameters:
    ----------
    data_dict : dict
        Dictionary containing user data, structured as {user_name: {'press_release': {win_type: data}}},
        where win_type is one of '00', '01', '10', '11' and data is a list of sensor readings.

    est : str, default='cov'
        Covariance estimation method. Options include 'cov' (sample covariance), 'lwf' (Ledoit-Wolf),
        'scm', 'oas', 'mcd', etc. Passed to pyriemann.estimation.Covariances.

    stat_type : str, default='arithmetic'
        Statistical type for calculations (currently unused).

    preprocessing_type : str or None, default=None
        Type of preprocessing to apply to the data (currently unused).

    apply_proj : bool, default=False
        Whether to project the matrices to tangent space (True) or return flattened lower
        triangular parts of the matrices (False).

    reference_type : str, default='identity'
        Reference point for tangent space projection:
        - 'identity': Uses identity matrix as reference
        - 'user_mean': Uses the mean of each user's covariance matrices
        - 'total_mean': Uses the global mean across all users' data

    list_of_channel_indices : list of lists, default=[[0, 1], [0, 2], [1, 2], [2, 1]]
        List of channel combinations to analyze. Each inner list specifies which channels to select.

    user_names : list or None, default=None
        List of user names to process. If None, processes all users in data_dict.

    metric : str, default='riemann'
        Riemannian metric to use for tangent space calculations. Options include 'riemann',
        'logeuclid', 'euclid', etc. Passed to pyriemann.tangentspace.TangentSpace.

    return_sequences : bool, default=False
        Whether to return sequences (currently unused).

    subwindow_size : int, default=90
        used only if return sequence. Eqch window framed into subwindows according to frame_step

    frame_step : int, default=8
        Step size for frame sliding (currently unused).

    Returns:
    -------
    dict
        Dictionary mapping combinations of user, channel indices, and window types to their
        corresponding matrices. Keys have the format 'user_{user_name}_{channel_indices}_{win_type}'.
        For each key, the value is either:
        - Tangent space representation of covariance matrices (if apply_proj=True)
        - Flattened lower triangular part of covariance matrices (if apply_proj=False)

    Notes:
    -----
    - The function first handles the reference point depending on reference_type:
      * For 'total_mean', it computes the mean across all users' data
      * For 'user_mean', it computes the mean for each individual user
      * For 'identity' (default), no explicit reference is computed (uses identity)

    - Data from each window type ('00', '01', '10', '11') is processed separately
    """



    if user_names is None:
        user_names = data_dict.keys()

    cov = Covariances(estimator=est)
    # Initialize and fit ts
    list_of_ts = []
    for _ in list_of_channel_indices:
        ts = TangentSpace(metric=metric)
        list_of_ts.append(ts)


    user_ch_win_type_csd_matrices_dict = {}
    if reference_type == 'total_mean':
        # collect all cov matrices
        all_covs = []
        for user_name in user_names:
            all_user_data = get_all_user_data(data_dict[user_name]['press_release']).numpy()
            covs_all_win_types = cov.transform(all_user_data)
            all_covs.append(covs_all_win_types)
        all_covs = tf.concat(all_covs, axis=0)
        # fit all ts
        for i, channel_indices in enumerate(list_of_channel_indices):
            ts = list_of_ts[i]
            all_covs_after_ch_selection = extract_channels_tf(all_covs, channel_indices).numpy()
            ts = ts.fit(all_covs_after_ch_selection)

    for user_name in user_names:
        all_user_data = get_all_user_data(data_dict[user_name]['press_release']).numpy()
        # get covariances for all window_types
        covs_all_win_types = cov.transform(all_user_data)
        for i, channel_indices in enumerate(list_of_channel_indices):
            ts = list_of_ts[i]
            # Fit ts
            covs_all_win_types_after_ch_selection = extract_channels_tf(covs_all_win_types, channel_indices).numpy()
            # by default ts.referense_ becomes identity in transform if it's not given before
            if reference_type == 'user_mean':
                ts = ts.fit(covs_all_win_types_after_ch_selection)

            for win_type in data_dict[user_name]['press_release'].keys():#['00', '01', '10', '11']:

                user_data = data_dict[user_name]['press_release'][win_type]
                user_data = tf.concat([tf.expand_dims(sensor_data, axis=-1) for sensor_data in user_data], axis=-1)
                user_data = tf.transpose(user_data, perm=(0,2,1)) #(batch_size, channels, eindow_size)
                if return_sequences:
                    user_data = tf.signal.frame(user_data, frame_length=subwindow_size, frame_step=frame_step,axis=-1) # (batch_size,channels,seq_len, subwindow_size)
                    user_data = tf.transpose(user_data, perm=(0,2,1,3))
                    train_data_np = user_data.numpy() if hasattr(user_data, 'numpy') else np.array(user_data)
                    covs = reshape_apply_reshape(train_data_np,cov)
                else:
                    train_data_np = user_data.numpy() if hasattr(user_data, 'numpy') else np.array(user_data)

                    covs = cov.transform(train_data_np)  # Covariance computation

                # print('ch', channel_indices)
                covs_after_ch_selection = extract_channels_tf(covs, channel_indices).numpy()

                if apply_proj:
                    if return_sequences:
                        result = reshape_apply_reshape(covs_after_ch_selection,ts)
                    else:
                        result = ts.transform(covs_after_ch_selection)
                else:
                    result = get_lower_trig_part_flattened(covs_after_ch_selection)
                user_ch_win_type_csd_matrices_dict[f'user_{user_name}_{channel_indices}_{win_type}'] = result
                # user_ch_win_type_csd_matrices_dict[
                #     f'{freq_id}_{i}_user_{user_name}_{channel_indices}_{win_type}_MEAN'] = riemann_mean

    # if plot:
    #     fig = interactive_plot_tensors_3d_improved(user_ch_win_type_csd_matrices_dict, title=title, output_html=f"{name}.html")
    #     fig.show()
    return user_ch_win_type_csd_matrices_dict

def process_folders(root_directory, window_size=128, pattern_len=128, additiona_folder='weight_estimation'):
    result_dict = {}

    # Walk through all directories and files
    for user_folder in os.listdir(root_directory):
        print(f'Processing user {user_folder}')
        user_path = os.path.join(root_directory, user_folder)

        # Check if it's a directory
        if os.path.isdir(user_path):
            if additiona_folder is not None:
                # Check for press_release folder
                addition_folder_path = os.path.join(user_path, additiona_folder)
                user_path = addition_folder_path

            # if os.path.isdir(press_release_path):
            # Go through each CSV file in press_release folder
            for file_name in os.listdir(user_path):
                if file_name.endswith('.csv'):
                    # Extract activity name (remove .csv extension)
                    activity = file_name.split('.')[0]

                    # Create the key: user_name_activity
                    key = f"{user_folder}_{activity}"

                    # Create full file path
                    full_file_path = os.path.join(user_path, file_name)

                    # Build dictionary entry
                    result_dict[key] = {
                        'press_release': build_windows_dict_from_file(
                            full_file_path,
                            window_size_snc=window_size,
                            pattern_len=pattern_len
                        )
                    }

    return result_dict


def extract_channels_tf(tensor, channel_indices):
    """
    Extract specific channels from a tensor of covariance matrices using TensorFlow.

    Parameters:
    -----------
    tensor : tf.Tensor
        Tensor of shape (..., ch, ch)
    channel_indices : list
        List of channel indices to extract

    Returns:
    --------
    extracted : tf.Tensor
        Tensor of shape (..., len(channel_indices), len(channel_indices))
    """
    # Convert channel indices to a tensor
    indices = tf.constant(channel_indices, dtype=tf.int32)

    # Use tf.gather to extract specified channels
    # First gather rows
    extracted = tf.gather(tensor, indices, axis=-2)
    # Then gather columns
    extracted = tf.gather(extracted, indices, axis=-1)

    return extracted

def get_windowed_data_dict(data_dict, window_size=128, frame_step=8):
    '''data_dict
      name: weight: list of dictionaries with keys dict_keys(['name', 'snc_1', 'snc_2', 'snc_3', 'file_path', 'contact', 'phase']) '''

    windowed_dict = {}
    for user, weight_dict in data_dict.items():
        windowed_dict[user] = {}
        for weight, data_list in weight_dict.items():
            framed_snc_list = []
            for data in data_list:
                data = pd.read_csv(data['file_path'])

                # Extract sensors
                snc1 = data['Snc1'].values
                snc2 = data['Snc2'].values
                snc3 = data['Snc3'].values

                # if apply_zpk2sos_filter:
                #     snc1 = zpk2sos_filter(snc1)
                #     snc2 = zpk2sos_filter(snc2)
                #     snc3 = zpk2sos_filter(snc3)

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
                framed_snc_list.append(framed_signal)

            framed_data = tf.concat(framed_snc_list,axis=0)
            windowed_dict[user][weight] = tf.transpose(framed_data, perm=(0,2,1))

    return windowed_dict






if __name__ == "__main__":
    # parameters
    window_size = 128#128
    pattern_len = 128#128
    reference_type='identity'

    # take dict for all usera
    user_names = None#['Daniel']
    weight_dir = '/home/wld-algo-6/Data/SortedCleaned'
    # file_dir = '/home/wld-algo-6/Data/Sorted'
    person_dict = get_weight_file_from_dir(weight_dir)
    persons_dict = {
        user: {weight: [dict for dict in list if dict['contact'] == 'M'] for weight, list in person_dict[user].items()}
        for user in person_dict.keys()}


    # data_dict_new = process_folders(weight_dir, window_size, pattern_len, additiona_folder='weight_estimation')
    windowed_dict = get_windowed_data_dict(persons_dict, window_size=128, frame_step=8)

    csd_layer = SequentialCrossSpectralDensityLayer_pyriemann(

        return_sequence=False,
        frame_length=128,
        frame_step=128,
        preprocessing_type=None,#'scattering',
        main_window_size=window_size,  # window size
        take_tangent_proj=True,
        base_point_calculation=reference_type,
    )
    num_of_components = 3

    # tangent_vectors = csd_layer(windowed_dict['Alisa'][0.0])
    tangent_dict = {user: {weight: csd_layer(windows) for weight, windows in weight_dict.items()} for user, weight_dict in windowed_dict.items()}
    tangent_dict_2= tangent_dict#{user: {weight: tsv[:, 0, :] for weight, tsv in weight_dict.items()} for user, weight_dict in
       # tangent_dict.items()}
    tangent_dict_3d , proj = apply_projection_to_dict(tangent_dict_2, n_components=num_of_components, perplexity=10,
                                               random_state=42,  # proj='pca',
                                               metric="euclidean", proj='pca', coord_0=5, coord_1=6)

    # tangent_dict_3d = {user:{weight: tsv[:,2,:3] for weight, tsv in weight_dict.items()} for user, weight_dict in tangent_dict.items()}

    dict_for_plot = {}
    for user, weight_dict in tangent_dict_3d.items():
        for weight, tensor in weight_dict.items():
            dict_for_plot[f'{user}_{weight}'] = tensor



    # # CSD SPACE
    fig = interactive_plot_tensors_3d_improved(dict_for_plot, title=f'spd_space_{reference_type}',
                                               output_html=f"{f'spd_space_{reference_type}'}.html")
    fig.show()
    # Then save using plotly's io functionality
    pio.write_html(fig, file='spd_space.html', auto_open=False)
    print("Figure saved using plotly.io.write_html")

    # visualizer = TensorVisualizer(user_ch_win_type_csd_matrices_dict)
    # visualizer.show()
