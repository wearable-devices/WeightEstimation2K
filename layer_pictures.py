import matplotlib.pyplot as plt
from tensorflow import float32

from custom.layers import ScatteringTimeDomain, SEMGScatteringTransform
from utils.get_data import get_windowed_data_from_file, get_weight_file_from_dir, mean_scattering_snc, \
    process_normal_csv
import  tensorflow as tf
from utils.maps import apply_projection_to_dict
from utils.plotting_functions import plot_movie_tensors, plot_heatmap_tensors, snc_scat_all, plot_feature_space


def get_scattering_transform(snc1_framed,snc2_framed,snc3_framed, type='old'):
    if type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif type == 'SEMG':
        scattering_layer = SEMGScatteringTransform()
    scattered_snc1, scattered_snc11 = scattering_layer(snc1_framed)
    scattered_snc1 = tf.squeeze(scattered_snc1, axis=-1)
    scattered_snc11 = tf.squeeze(scattered_snc11, axis=-1)

    scattered_snc2, scattered_snc22 = scattering_layer(snc2_framed)
    scattered_snc2 = tf.squeeze(scattered_snc2, axis=-1)
    scattered_snc22 = tf.squeeze(scattered_snc22, axis=-1)

    scattered_snc3, scattered_snc33 = scattering_layer(snc3_framed)
    scattered_snc3 = tf.squeeze(scattered_snc3, axis=-1)
    scattered_snc33 = tf.squeeze(scattered_snc33, axis=-1)



    return scattered_snc1, scattered_snc11, scattered_snc2, scattered_snc22, scattered_snc3, scattered_snc33

if __name__ == "__main__":
    SENSOR_NUM = 2
    file_path = '/home/wld-algo-6/DataCollection/Data/Daniel/Train/Daniel_1_weight_500_0_Leaning_M.csv'
    window_size_snc = 512
    snc1_framed, snc2_framed, snc3_framed = get_windowed_data_from_file(file_path, window_size_snc)
    snc1_data, snc2_data, snc3_data = process_normal_csv(file_path)
    snc1_data = tf.cast(snc1_data, dtype=float32) / 2 ** 23
    snc2_data = tf.cast(snc2_data, dtype=float32) / 2 ** 23
    snc3_data = tf.cast(snc3_data, dtype=float32) / 2 ** 23
    J_snc = 7
    Q_snc =(2, 1)
    undersampling = 1
    snc1_framed = tf.cast(snc1_framed, dtype=float32)/2**23
    snc2_framed = tf.cast(snc2_framed, dtype=float32)/2**23
    snc3_framed = tf.cast(snc3_framed, dtype=float32)/2**23

    sensors_framed = [snc1_framed, snc2_framed, snc3_framed]
    snc_framed = sensors_framed[SENSOR_NUM-1]
    scattered_snc1, scattered_snc11, scattered_snc2, scattered_snc22, scattered_snc3, scattered_snc33 = get_scattering_transform(snc1_framed,snc2_framed,snc3_framed)

    tensors_for_movie = [snc1_framed, scattered_snc1, scattered_snc11]
    names = [f'snc{1}_framed', 'scattered_1', 'scattered_11']

    # plot_heatmap_tensors([snc1_data, scattered_snc1[0], scattered_snc11[0]], names=names, title=None, normalize=False, save_path=None)
    # plot_movie_tensors(tensors_for_movie, names = names, im_min =None, im_max=None)
    # snc_scat_all(snc1_data, snc2_data, snc3_data, scattered_snc1, scattered_snc2, scattered_snc3,
    #           scattered_snc11, scattered_snc22, scattered_snc33)

    scattered_snc1_mean =  tf.reduce_mean(scattered_snc1, axis=-1)
    persons_dict = get_weight_file_from_dir('/home/wld-algo-6/DataCollection/Data')
    output_dict = mean_scattering_snc(persons_dict, window_size=162, samples_per_weight_per_person=25, scattering_type='old')
    proj_dict  = apply_projection_to_dict(output_dict, n_components=3, perplexity=10, random_state=42, proj='pca',
                             metric="euclidean")

    plot_feature_space(proj_dict, num_of_components=3)

    ttt=1


