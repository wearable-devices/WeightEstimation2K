import os
import tensorflow as tf
import numpy as np
import plotly.graph_objects as go
from filterpy.kalman import predict
from numpy.core.records import record
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import keract
from tensorflow import float32
from tensorflow.keras.callbacks import ModelCheckpoint
import plotly.io as pio
from tensorflow.python.ops.losses.losses_impl import mean_squared_error
# cursor suggestions
from tensorflow.python.ops.losses.losses_impl import mean_squared_error

from PIL import Image
import io
import keras

import matplotlib.pyplot as plt

# from stam import data_mode
import keras.ops as K

# from stam import output_dict
from utils.maps import filter_dict_by_keys

class SaveKerasModelCallback(keras.callbacks.Callback):
    def __init__(self, save_path, model_name, phase='train'):
        super(SaveKerasModelCallback, self).__init__()
        self.save_path = save_path
        self.model_name = model_name
        self.phase= phase
        os.makedirs(save_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if self.phase=='test':
            self.model.save(os.path.join(self.save_path, self.model_name + str(epoch) + '.keras'), save_format='keras')
            print(f'Model saved to {self.save_path}')

    def on_train_end(self, logs=None):
        self.model.save(os.path.join(self.save_path, self.model_name + '.keras'), save_format='keras')
        print(f'Model saved to {self.save_path}')


class FeatureSpacePlotCallback(keras.callbacks.Callback):
    def __init__(self, persons_dict, trial_dir, layer_name, used_persons='all', proj='pca', metric="euclidean",
                 num_of_components='3', considered_weights=[0, 0.5, 1,2], data_mode='all',
                 samples_per_label_per_person=10, picture_name_prefix='name', phase='train'):
        super(FeatureSpacePlotCallback, self).__init__()

        if used_persons == 'all':
            self.persons_dict = persons_dict
        else:
            self.persons_dict = filter_dict_by_keys(persons_dict, used_persons)
        self.trial_dir = trial_dir
        self.layer_name = layer_name
        self.proj = proj
        self.metric = metric
        self.phase = phase
        self.data_mode = data_mode
        self.num_of_components = num_of_components
        self.picture_name_prefix = picture_name_prefix
        self.picture_name = picture_name_prefix
        # self.persons_dict_labeled = persons_dict_labeled
        self.considered_weights = considered_weights
        # if persons_dict_labeled is None:
        #     self.persons_dict_labeled = self.persons_dict
        self.samples_per_label_per_person = samples_per_label_per_person
        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        # Use epoch number here to create the picture name
        self.picture_name = f"{self.picture_name_prefix}{epoch}"

    def on_train_end(self, logs=None):
        # try:
        self.calc_feature_space()
        self.plot_feature_space_optuna()

        # except Exception as e:
        #     print('Could not plot feature space!')
        #     print('layer: ', self.layer_name)
        #     print(' picture name: ', self.picture_name)

    def on_test_end(self,epoch, logs=None):
        if self.phase == 'test' and  self.current_epoch % 5  == 0:
            self.calc_feature_space()
            self.plot_feature_space_optuna()

    def calc_feature_space(self):
        """ Calculate and project the feature space from model layer outputs for each person and label.  """

        window_size = self.model.input['snc_1'].shape[-1]

        if len(self.model.inputs) == 7:  # model has a labeled input
            samples_per_weight = int(self.model.inputs[3].shape[
                                         1] / 6)  # 6 = number of different weights whitch supposed to be the same for all persons
            labeled_input_dict = get_labeled_tensors(self.persons_dict_labeled, samples_per_weight=samples_per_weight,
                                                     window_size=window_size)

        layer_output_dict = {person: {} for person in
                             self.persons_dict.keys()}  # {person: {weight: [] for weight in persons_dict[person]} for person in persons_dict.keys()}

        for person, weight_dict in self.persons_dict.items():
            weight_dict_part = {weight: weight_dict[weight] for weight in self.considered_weights if
                                weight in weight_dict}
            # label = weight
            for label, records in weight_dict_part.items():
                # tf.print('recored', records)
                # tf.print('data mode', self.data_mode)
                if self.data_mode == 'all':
                    used_records = records
                else:
                    used_records = [record for record in records if record['phase'] == self.data_mode]
                # tf.print('used_recored', used_records)
                if len(used_records)>0:
                    snc1_batch = []
                    snc2_batch = []
                    snc3_batch = []
                    for _ in range(self.samples_per_label_per_person):
                        # Randomly select a file for this label
                        file_idx = tf.random.uniform([], 0, len(records), dtype=tf.int32)
                        file_data = records[file_idx.numpy()]
                        # Generate a random starting point within the file
                        start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - window_size + 1,
                                                  dtype=tf.int32)
                        # Extract the window
                        snc1_batch.append(file_data['snc_1'][start:start + window_size])
                        snc2_batch.append(file_data['snc_2'][start:start + window_size])
                        snc3_batch.append(file_data['snc_3'][start:start + window_size])
                        # labels.append(label)

                    persons_input_data = [tf.stack(snc1_batch), tf.stack(snc2_batch), tf.stack(snc3_batch)]
                    if len(self.model.inputs) == 7:
                        persons_input_data = [tf.expand_dims(snc_data, axis=1) for snc_data in persons_input_data]
                        persons_labeled_input_dict = labeled_input_dict[person]
                        persons_labeled_input_data = [tf.repeat(tensor[tf.newaxis, :, :], repeats=len(snc1_batch), axis=0)
                                                      for tensor in persons_labeled_input_dict.values()]
                        persons_input_data = persons_input_data + persons_labeled_input_data

                    layer_predictions = get_layer_output(self.model, persons_input_data, self.layer_name)
                    # tf.print('layer_pred', layer_predictions)
                    # tf.print('actiVAT',keract.get_activations(self.model, persons_input_data, layer_names=[self.layer_name]))
                    layer_predictions = keras.layers.Flatten()(layer_predictions)
                    layer_output_dict[person][label] = layer_predictions
        # Extract features from an intermediate layer and project them
        # tf.print('layer output', layer_output_dict)
        self.projected_layer_output_dict = apply_projection_to_dict(layer_output_dict,
                                                                    n_components=self.num_of_components,
                                                                    perplexity=30, random_state=42, proj=self.proj,
                                                                    metric=self.metric)
        print(f'Feature space for layer {self.layer_name} is calculated')
        # print(self.projected_layer_output_dict)
        # except:
        #     print('Problem with layer', self.layer_name)

    def plot_feature_space_optuna(self):
        # Color map for labels
        color_map = {
            0: 'red',
            0.5: 'pink',
            1: 'blue',
            2: 'green',
            4: 'yellow',
            6: 'brown',
            8: 'black'
        }

        color_map_2 = {
            0: 'crimson',
            0.5: 'pink',
            1: 'navy',
            2: 'forestgreen',  # Changed from 'forest_green' to 'forestgreen'
            4: 'gold',
            6: 'sienna',
            8: 'dimgray'  # Changed from 'charcoal' to 'dimgray' as charcoal isn't a standard CSS color
        }

        # Marker map for persons
        markers = ['square']  # , 'square-open', 'circle','square', 'diamond', 'cross', 'x', 'diamond-open']
        marker_map = {person: markers[i % len(markers)] for i, person in enumerate(self.persons_dict.keys())}

        # Create the plot
        fig = go.Figure()

        user_num = 0
        for person, labels in self.projected_layer_output_dict.items():
            user_num += 1
            used_color_map = color_map_2 if user_num % 2 == 0 else color_map
            for label, tensor in labels.items():
                if self.num_of_components == 2:
                    fig.add_trace(go.Scatter(
                        x=tensor[:, 0],
                        y=tensor[:, 1],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=used_color_map[label],
                            symbol=marker_map[person],
                            opacity=0.8
                        ),
                        name=f"{person} - {label}"
                    ))
                else:
                    fig.add_trace(go.Scatter3d(
                        x=tensor[:, 0],
                        y=tensor[:, 1],
                        z=tensor[:, 2],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=used_color_map[label],
                            symbol=marker_map[person],
                            opacity=0.8
                        ),
                        # name=f"{person} - {label}"
                        name=f" {person} - {label} kg"
                    ))

        # Update layout
        fig.update_layout(
            title=f"Feature Space Visualization",  # of {self.layer_name}_proj_{self.proj}_name_{self.picture_name}",
            xaxis_title="X",
            yaxis_title="Y",
            legend_title="User - Weight"
        )

        # Save the plot as HTML
        html_path = os.path.join(self.trial_dir,
                                 f"{self.layer_name}_{self.proj}_{self.metric}_{self.num_of_components}_comp_feature_space_{self.picture_name}.html")
        fig.write_html(html_path, full_html=True)
        # Save the plot as a GIF
        gif_path = os.path.join(self.trial_dir,
                                f"{self.layer_name}_{self.proj}_{self.metric}_{self.num_of_components}_comp_feature_space_{self.picture_name}.gif")

        # Create frames for animation
        frames = []
        for i in range(0, 360, 10):  # Rotate 360 degrees in steps of 10
            fig.update_layout(scene_camera=dict(eye=dict(x=1.25 * np.cos(np.radians(i)),
                                                         y=1.25 * np.sin(np.radians(i)),
                                                         z=0.5)))
            img_bytes = fig.to_image(format="png")
            img = Image.open(io.BytesIO(img_bytes))
            frames.append(img)

        # Save frames as GIF
        gif_path = os.path.join(self.trial_dir,
                                f"{self.layer_name}_{self.proj}_{self.metric}_{self.num_of_components}_comp_feature_space_{self.picture_name}.gif")

        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=200, loop=0)

        print(f"Animated GIF saved to: {gif_path}")

        print(f"Interactive plot saved to: {html_path}")

        # # Optionally, save as PNG as well
        # png_path = os.path.join(self.trial_dir,
        #                         f"{self.layer_name}_{self.proj}_{self.metric}_{self.num_of_components}_comp_feature_space_{self.picture_name}.png")
        # fig.write_image(png_path)
        # print(f"Static image saved to: {png_path}")

    def cleanup(self):
        self.projected_layer_output_dict = None
        keras.backend.clear_session()


class OutputPlotCallback(keras.callbacks.Callback):
    def __init__(self, persons_dict, trial_dir, #persons_dict_labeled=None,
                 samples_per_label_per_person=10, output_num=0, used_persons='all',data_mode='all', picture_name='name', phase='train'):
        super(OutputPlotCallback, self).__init__()
        '''used_persons - list of persons to work with'''

        if used_persons == 'all':
            self.persons_dict = persons_dict
        else:
            self.persons_dict = filter_dict_by_keys(persons_dict, used_persons)
        self.trial_dir = trial_dir
        self.phase = phase
        self.data_mode = data_mode
        self.picture_name = picture_name
        self.output_num = output_num
        self.samples_per_label_per_person = samples_per_label_per_person

        self.output_dict = {person: {} for person in
                            self.persons_dict.keys()}  # {person: {weight: [] for weight in persons_dict[person]} for person in persons_dict.keys()}
        self.personal_accuracy = {}

        '''if model supposed to have labeled input it will be taken from self.persons_dict_labeled'''

    def on_train_end(self, logs=None):
        try:
            self.calc_feature_space()
        except:
            print('Could not calc  output Callback')
        try:
            self.plot_feature_space_optuna()
            # print(self.personal_accuracy)
        except:
            print('Could not  plot output Callback')

    def on_test_end(self, logs=None):
        if self.phase == 'test':
            self.calc_feature_space()
            self.plot_feature_space_optuna()

    # def get_second_stage_accuracy(self):
    #     # Check if self.output_dict[first key in the dict] is empty
    #     # personal_accuracy = {}
    #     first_person = next(iter(self.output_dict))
    #     if not self.output_dict[first_person]:
    #         print(f"Output dict for {first_person} is empty. Calculating feature space...")
    #         self.calc_feature_space()
    #     else:
    #         print(f"Output dict for {first_person}m is not empty. Proceeding with existing data.")
    #     for person, weight_dict in self.output_dict.items():
    #         personal_errors = {weight: (tf.reduce_mean(self.output_dict[person][weight][0]) - weight).numpy() for weight
    #                            in weight_dict.keys()}
    #         mean_squared_error = np.sqrt(np.mean([error ** 2 for error in personal_errors.values()]))
    #         personal_errors['mean_squared_error'] = mean_squared_error
    #         self.personal_accuracy[person] = personal_errors

    def calc_feature_space(self):
        """this method  """
        window_size = self.model.input['snc_1'].shape[-1]

        if len(self.model.inputs) == 7:  # model has a labeled input
            samples_per_weight = int(self.model.inputs[3].shape[
                                         1] / 6)  # 6 = number of different weights whitch supposed to be the same for all persons
            labeled_input_dict = get_labeled_tensors(self.persons_dict_labeled, samples_per_weight=samples_per_weight,
                                                     window_size=window_size)
        for person, weight_dict in self.persons_dict.items():

            for weight, records in weight_dict.items():
                if self.data_mode == 'all':
                    used_records = records
                else:
                    used_records = [record for record in records if record['phase'] == self.data_mode]
                if len(used_records)>0:
                    snc1_batch = []
                    snc2_batch = []
                    snc3_batch = []
                    for _ in range(self.samples_per_label_per_person):
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
                    if isinstance(self.model.input, dict):
                        persons_input_data = {'snc_1': tf.cast(tf.stack(snc1_batch), dtype=float32),
                                              'snc_2': tf.cast(tf.stack(snc2_batch), dtype=float32),
                                              'snc_3': tf.cast(tf.stack(snc3_batch), dtype=float32)}
                    else:
                        persons_input_data = [tf.stack(snc1_batch), tf.stack(snc2_batch), tf.stack(snc3_batch)]
                    if len(self.model.inputs) == 7:
                        persons_input_data = [tf.expand_dims(snc_data, axis=1) for snc_data in persons_input_data]
                        persons_labeled_input_dict = labeled_input_dict[person]
                        persons_labeled_input_data = [tf.repeat(tensor[tf.newaxis, :, :], repeats=len(snc1_batch), axis=0)
                                                      for tensor in persons_labeled_input_dict.values()]
                        persons_input_data = persons_input_data + persons_labeled_input_data


                    try:
                        predictions = self.model.predict(persons_input_data)
                        if isinstance(predictions, list):
                            predictions = predictions[self.output_num]
                    except:
                        predictions = self.model([persons_input_data])['weight_output']
                        # predictions = self.model([persons_input_data])['gaussian_output']
                    # predictions = tf.squeeze(tf.keras.layers.Flatten()(predictions), axis=-1)
                    self.output_dict[person][weight] = predictions
                # tf.print('weight', weight)
                #
                # tf.print('predictions', predictions.shape, predictions)
                # tf.print('output_dict', self.output_dict)

    def plot_feature_space_optuna(self):
        # Color map for labels
        color_map = {
            0: 'red',
            0.5: 'pink',
            1: 'blue',
            2: 'green',
            4: 'yellow',
            6: 'brown',
            8: 'black'
            # Add more colors for additional labels
        }

        # Marker map for persons
        markers = ['circle', 'square', 'diamond', 'cross', 'x', 'square-open', 'diamond-open']
        marker_map = {person: markers[i % len(markers)] for i, person in enumerate(self.persons_dict.keys())}

        # Create the plot
        fig = go.Figure()

        for person, labels in self.output_dict.items():
            # Dictionary to store mean values for each label
            mean_values = {}
            for label, tensor in labels.items():
                fig.add_trace(go.Scatter(
                    x=[label] * tensor.shape[0],  # Repeat the label for each point
                    y=tensor[:, 0],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color_map[label],
                        symbol=marker_map[person],
                        opacity=0.8
                    ),
                    name=f"{person} - {label}"
                ))

                # mean_values[label] = tf.reduce_mean(tensor[:, 0]).numpy()
                mean_values[label] = tf.reduce_mean(tensor).numpy()

            # Add line connecting mean points
            sorted_labels = sorted(mean_values.keys())
            fig.add_trace(go.Scatter(
                x=sorted_labels,
                y=[mean_values[label] for label in sorted_labels],
                mode='lines+markers',
                # line=dict(color='green', width=2),
                line=dict(color='green', dash='dash'),
                marker=dict(size=10, symbol='star'),
                name='Mean Values'
            ))
        # Add diagonal line y=x
        fig.add_trace(go.Scatter(
            x=[-0.1, 3],
            y=[-0.1, 3],
            mode='lines',
            line=dict(color='gray', dash='dash'),
            name='y=x'
        ))
        # Update layout
        fig.update_layout(
            title=f"Feature Space Visualization for {self.picture_name}",
            xaxis_title="Weight",
            yaxis_title="Predicted value",
            legend_title="Person - Label",
            xaxis=dict(
                range=[-0.1, 3],  # Set x-axis limits
                tickmode='array',
                tickvals=list(set([label for labels in self.output_dict.values() for label in labels.keys()])),
                ticktext=[f'{label} kg' for label in
                          set([label for labels in self.output_dict.values() for label in labels.keys()])]
            ),
            yaxis=dict(
                range=[-0.1, 3]  # Set y-axis limits
            )
        )

        # Save the plot as HTML
        html_path = os.path.join(self.trial_dir,
                                 f"{self.picture_name}_output.html")
        fig.write_html(html_path, full_html=True)
        print(f"Interactive plot saved to: {html_path}")

        # # Optionally, save as PNG as well
        # png_path = os.path.join(self.trial_dir,
        #                         f"{self.picture_name}_output.png")
        # fig.write_image(png_path)
        # print(f"Static image saved to: {png_path}")

    def cleanup(self):
        self.projected_layer_output_dict = None
        keras.backend.clear_session()


from custom.layers import ScatteringTimeDomain
from custom.layers import OrderedAttention
# from db_generators.get_db import get_weight_file_from_dir


# def ex_get_layer_predictions(model, input_data, layer_name):
#     '''returns layer output'''
#     try:
#         # tf.compat.v1.disable_eager_execution()
#         # activations = keract.get_activations(model, [input_data])
#         try:
#             activations = keract.get_activations(model, input_data, layer_names=[layer_name])
#         except:
#             activations = keract.get_activations(model, input_data)
#         # layer_extraction = activations[layer_name]
#         # Find the first layer name that includes the pattern
#         matching_layer = next((name for name in activations.keys() if layer_name in name), None)
#         if matching_layer is None:
#             raise ValueError(f"No layer found with name pattern: {layer_name}")
#
#         layer_extraction = activations[matching_layer]
#         if isinstance(layer_extraction, list) or isinstance(layer_extraction, tuple):
#             layer_extraction = layer_extraction[0]
#     except:
#         print(f'Problem with layer {layer_name}')
#         layer_extraction = 0
#     return layer_extraction


def get_layer_output(model, input_data, layer_name):
    intermediate_model = keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    return intermediate_model.predict(input_data)

def apply_projection_to_dict(input_dict, n_components=2, perplexity=10, random_state=42, proj='tsne',
                             metric="euclidean"):
    """takes an input_dict
        {person:{label:tensor_of_predictions}}
        tensor of predictions is of shape (batch_size,smth)
        and takes a projection of all predictions and gets a new dictionary with
        {person:{label:projected_predictions}}"""
    # Collect all predictions into a single array
    all_predictions = []
    for person, label_dict in input_dict.items():
        for label, predictions in label_dict.items():
            all_predictions.append(predictions.numpy())

    # Concatenate all predictions
    # tf.print('all_predictions', all_predictions)
    all_predictions = np.concatenate(all_predictions, axis=0)

    # Apply t-SNE
    if proj == 'tsne':
        projection = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state, metric=metric)
    elif proj == 'pca':
        projection = PCA(n_components=n_components)
    # tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
    if proj == 'none':
        projected_predictions = all_predictions[:, :n_components]
    else:
        projected_predictions = projection.fit_transform(all_predictions)

    # Create new dictionary with projected predictions
    new_dict = {}
    start_idx = 0
    for person, label_dict in input_dict.items():
        new_dict[person] = {}
        for label, predictions in label_dict.items():
            end_idx = start_idx + predictions.shape[0]
            new_dict[person][label] = tf.convert_to_tensor(projected_predictions[start_idx:end_idx], dtype=tf.float32)
            start_idx = end_idx

    return new_dict


def get_labeled_tensors(persons_dict, samples_per_weight=10, window_size=306):
    '''returns dict {person:{labeled_inputs}'''
    # create a labeled input
    labeled_input_dict = {}
    for person, weight_dict in persons_dict.items():
        person_weights = list(set(weight_dict))
        if len(person_weights) < 6:
            print(f'person {person} has not all data')
        labeled_snc1_list, labeled_snc2_list, labeled_snc3_list = [], [], []
        person_support_labels = []
        for label in sorted(person_weights):
            labeled_snc_1, labeled_snc_2, labeled_snc_3 = [], [], []
            person_label_data = weight_dict[label]
            file_indices = np.random.choice(len(person_label_data), samples_per_weight, replace=True)
            for i, file_idx in enumerate(file_indices):
                file_data = person_label_data[file_idx]
                start = np.random.randint(0, len(file_data['snc_1']) - window_size + 1)
                snc1 = file_data['snc_1'][start:start + window_size]
                snc2 = file_data['snc_2'][start:start + window_size]
                snc3 = file_data['snc_3'][start:start + window_size]

                # if i < self.sampels_per_weight:
                labeled_snc_1.append(tf.expand_dims(snc1, axis=0))
                labeled_snc_2.append(tf.expand_dims(snc2, axis=0))
                labeled_snc_3.append(tf.expand_dims(snc3, axis=0))
            labeled_snc_1_with_same_label = tf.concat(labeled_snc_1, axis=0)
            labeled_snc_2_with_same_label = tf.concat(labeled_snc_2, axis=0)
            labeled_snc_3_with_same_label = tf.concat(labeled_snc_3, axis=0)
            labeled_snc1_list.append(labeled_snc_1_with_same_label)
            labeled_snc2_list.append(labeled_snc_2_with_same_label)
            labeled_snc3_list.append(labeled_snc_3_with_same_label)
            person_support_labels.extend([label] * samples_per_weight)

        person_support_labels = tf.expand_dims(tf.convert_to_tensor(person_support_labels), axis=-1)
        labeled_snc1 = tf.concat(labeled_snc1_list, axis=0)
        labeled_snc2 = tf.concat(labeled_snc2_list, axis=0)
        labeled_snc3 = tf.concat(labeled_snc3_list, axis=0)

        labeled_input_dict[person] = {'labeled_snc1_tensor': labeled_snc1,
                                      'labeled_snc2_tensor': labeled_snc2,
                                      'labeled_snc3_tensor': labeled_snc3,
                                      'person_support_labels': person_support_labels}
    return labeled_input_dict


def get_labeled_list(persons_dict, samples_per_weight=10, window_size=306):
    '''returns dict {person:{label: inputs like a list}'''
    # create a labeled input
    labeled_input_dict = {}
    for person, weight_dict in persons_dict.items():
        labeled_input_dict[person] = {}
        person_weights = list(set(weight_dict))
        if len(person_weights) < 6:
            print(f'person {person} has not all data')
        for label in sorted(person_weights):
            labeled_input_dict[person][label] = {'snc_1': [], 'snc_2': [],
                                                 'snc_3': []}
            labeled_snc_1, labeled_snc_2, labeled_snc_3 = [], [], []
            person_label_data = weight_dict[label]
            file_indices = np.random.choice(len(person_label_data), samples_per_weight, replace=True)
            for i, file_idx in enumerate(file_indices):
                file_data = person_label_data[file_idx]
                start = np.random.randint(0, len(file_data['snc_1']) - window_size + 1)
                snc1 = file_data['snc_1'][start:start + window_size]
                snc2 = file_data['snc_2'][start:start + window_size]
                snc3 = file_data['snc_3'][start:start + window_size]

                labeled_input_dict[person][label]['snc_1'].append(tf.expand_dims(snc1, axis=0))
                labeled_input_dict[person][label]['snc_2'].append(tf.expand_dims(snc2, axis=0))
                labeled_input_dict[person][label]['snc_3'].append(tf.expand_dims(snc3, axis=0))
            labeled_input_dict[person][label]['snc_1'] = tf.concat(labeled_input_dict[person][label]['snc_1'], axis=0)
            labeled_input_dict[person][label]['snc_2'] = tf.concat(labeled_input_dict[person][label]['snc_2'], axis=0)
            labeled_input_dict[person][label]['snc_3'] = tf.concat(labeled_input_dict[person][label]['snc_3'], axis=0)
    return labeled_input_dict


class Output_2d_PlotCallback(keras.callbacks.Callback):
    def __init__(self, persons_dict, trial_dir, #persons_dict_labeled=None,
                 samples_per_label_per_person=10,used_persons='all',data_mode='all', picture_name='name', phase='train'):
        super(Output_2d_PlotCallback, self).__init__()
        '''used_persons - list of persons to work with'''

        if used_persons == 'all':
            self.persons_dict = persons_dict
        else:
            self.persons_dict = filter_dict_by_keys(persons_dict, used_persons)
        self.trial_dir = trial_dir
        self.phase = phase
        self.data_mode = data_mode
        self.picture_name = picture_name
        self.samples_per_label_per_person = samples_per_label_per_person

        self.output_dict = {person: {} for person in
                            self.persons_dict.keys()}  # {person: {weight: [] for weight in persons_dict[person]} for person in persons_dict.keys()}
        # self.personal_accuracy = {}

        '''if model supposed to have labeled input it will be taken from self.persons_dict_labeled'''

    def on_train_end(self, logs=None):
        try:
            self.calc_feature_space()
        except:
            print('Could not calc  output Callback')
        try:
            self.plot_feature_space_optuna()
            # print(self.personal_accuracy)
        except:
            print('Could not  plot output Callback')

    def on_test_end(self, logs=None):
        if self.phase == 'test':
            self.calc_feature_space()
            self.plot_feature_space_optuna()

    # def get_second_stage_accuracy(self):
    #     # Check if self.output_dict[first key in the dict] is empty
    #     # personal_accuracy = {}
    #     first_person = next(iter(self.output_dict))
    #     if not self.output_dict[first_person]:
    #         print(f"Output dict for {first_person} is empty. Calculating feature space...")
    #         self.calc_feature_space()
    #     else:
    #         print(f"Output dict for {first_person}m is not empty. Proceeding with existing data.")
    #     for person, weight_dict in self.output_dict.items():
    #         personal_errors = {weight: (tf.reduce_mean(self.output_dict[person][weight][0]) - weight).numpy() for weight
    #                            in weight_dict.keys()}
    #         mean_squared_error = np.sqrt(np.mean([error ** 2 for error in personal_errors.values()]))
    #         personal_errors['mean_squared_error'] = mean_squared_error
    #         self.personal_accuracy[person] = personal_errors

    def calc_feature_space(self):
        """this method  """
        window_size = self.model.input['snc_1'].shape[-1]

        if len(self.model.inputs) == 7:  # model has a labeled input
            samples_per_weight = int(self.model.inputs[3].shape[
                                         1] / 6)  # 6 = number of different weights whitch supposed to be the same for all persons
            labeled_input_dict = get_labeled_tensors(self.persons_dict_labeled, samples_per_weight=samples_per_weight,
                                                     window_size=window_size)
        for person, weight_dict in self.persons_dict.items():

            for weight, records in weight_dict.items():
                if self.data_mode == 'all':
                    used_records = records
                else:
                    used_records = [record for record in records if record['phase'] == self.data_mode]
                if len(used_records)>0:
                    snc1_batch = []
                    snc2_batch = []
                    snc3_batch = []
                    for _ in range(self.samples_per_label_per_person):
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
                    if len(self.model.inputs) == 7:
                        persons_input_data = [tf.expand_dims(snc_data, axis=1) for snc_data in persons_input_data]
                        persons_labeled_input_dict = labeled_input_dict[person]
                        persons_labeled_input_data = [tf.repeat(tensor[tf.newaxis, :, :], repeats=len(snc1_batch), axis=0)
                                                      for tensor in persons_labeled_input_dict.values()]
                        persons_input_data = persons_input_data + persons_labeled_input_data
                    try:
                        predictions = self.model([persons_input_data])[0]
                    except:
                        predictions = self.model([persons_input_data])['gaussian_output']
                    # predictions = tf.squeeze(tf.keras.layers.Flatten()(predictions), axis=-1)
                    self.output_dict[person][weight] = predictions
                # tf.print('weight', weight)
                #
                # tf.print('predictions', predictions.shape, predictions)
                # tf.print('output_dict', self.output_dict)

    def plot_feature_space_optuna(self):
        # Color map for labels
        color_map = {
            0: 'red',
            0.5: 'pink',
            1: 'blue',
            2: 'green',
            4: 'yellow',
            6: 'brown',
            8: 'black'
            # Add more colors for additional labels
        }

        # Marker map for persons
        markers = ['circle', 'square', 'diamond', 'cross', 'x', 'square-open', 'diamond-open']
        marker_map = {person: markers[i % len(markers)] for i, person in enumerate(self.persons_dict.keys())}

        # Create the plot
        fig = go.Figure()

        for person, labels in self.output_dict.items():
            # Dictionary to store mean values for each label
            mean_values = {}
            for label, tensor in labels.items():
                fig.add_trace(go.Scatter(
                    x=tensor[:, 0],  #  predicted weight
                    y=tensor[:, 1],  # sigma
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=color_map[label],
                        symbol=marker_map[person],
                        opacity=0.8
                    ),
                    name=f"{person} - {label}"
                ))

                # mean_values[label] = tf.reduce_mean(tensor[:, 0]).numpy()



        # Update layout
        fig.update_layout(
            title=f"Feature Space Visualization for {self.picture_name}",
            xaxis_title="Weight",
            yaxis_title="Predicted value",
            legend_title="Person - Label",
            xaxis=dict(
                range=[-0.1, 3],  # Set x-axis limits
                tickmode='array',
                tickvals=list(set([label for labels in self.output_dict.values() for label in labels.keys()])),
                ticktext=[f'{label} kg' for label in
                          set([label for labels in self.output_dict.values() for label in labels.keys()])]
            ),
            yaxis=dict(
                range=[-0.1, 3]  # Set y-axis limits
            )
        )

        # Save the plot as HTML
        html_path = os.path.join(self.trial_dir,
                                 f"{self.picture_name}_output.html")
        fig.write_html(html_path, full_html=True)
        print(f"Interactive plot saved to: {html_path}")

        # # Optionally, save as PNG as well
        # png_path = os.path.join(self.trial_dir,
        #                         f"{self.picture_name}_output.png")
        # fig.write_image(png_path)
        # print(f"Static image saved to: {png_path}")

    def cleanup(self):
        self.projected_layer_output_dict = None
        keras.backend.clear_session()

class MetricsTrackingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"\nEpoch {epoch} logs:", logs)
        if not logs:
            print("WARNING: logs dictionary is empty!")
        if not hasattr(self.model, 'history') or not self.model.history:
            print("WARNING: model history is not initialized!")


if __name__ == "__main__":
    # model_snc_path = '/home/wld-algo-5/Production/WeightEstimation/logs/13-08-2024-15-46-34/trials/trial_0/model_trial_00.keras'
    # custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain, 'OrderedAttention': OrderedAttention}
    # model = tf.keras.models.load_model(model_snc_path, custom_objects=custom_objects, compile=False)
    #
    train_dir = '/home/wld-algo-5/Data/17_7_2024 data base/Train'
    # test_dir = '/home/wld-algo-5/Data/17_7_2024 data base/Test'
    #
    train_persons_dict = get_weight_file_from_dir(train_dir)
    dict = get_labeled_list(train_persons_dict, samples_per_weight=10, window_size=306)

    emb_map_path = '/home/wld-algo-5/Production/WeightEstimation/logs/25-08-2024-10-47-01/trials/trial_0/emb_map_trial_0.keras'
    model = keras.models.load_model(emb_map_path,  # custom_objects=custom_objects,
                                       compile=False)

