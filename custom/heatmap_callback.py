import keras
from custom.layers import SEMGScatteringTransform
import tensorflow as tf
# from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from utils.get_data import get_weight_file_from_dir


def plot_heat_tensor(heat_tensor, point_tensors=None, colors=None, title='Heatmap',
                     x_min=-48, x_max=48, step=0.5,  # Add coordinate bounds
                     column_index='Column Index', row_index='Row Index', plot_grid=True):
    # Convert heat tensor to numpy if needed
    if tf.is_tensor(heat_tensor):
        data = heat_tensor.numpy()
    else:
        data = heat_tensor

    # Create coordinate arrays
    x_coords = np.arange(x_min, x_max, step)
    y_coords = np.arange(x_min, x_max, step)

    # Create the figure and plot heatmap
    plt.figure(figsize=(10, 8))
    # sns.heatmap(data, cmap='viridis', annot=False)
    # Plot heatmap with extent to specify the coordinate ranges
    im = plt.imshow(data, extent=[x_min, x_max, x_min, x_max],
               origin='lower', cmap='viridis')

    # If point tensors are provided, plot them
    if point_tensors is not None:
        # If colors not provided, generate them
        if colors is None:
            colors = plt.cm.rainbow(np.linspace(0, 1, len(point_tensors)))

        # Plot each tensor's points directly using their coordinates
        for i, (tensor, color) in enumerate(zip(point_tensors, colors)):
            if tf.is_tensor(tensor):
                points = tensor.numpy()
            else:
                points = tensor

            # Plot points using their actual coordinates
            plt.scatter(points[:, 1], points[:, 0], c=[color],
                        marker='x', s=50, alpha=0.7, label=f'Points {i + 1}')

    plt.title(title)
    plt.xlabel(column_index)
    plt.ylabel(row_index)

    # Add grid if requested
    if plot_grid:
        plt.grid(True)

    # Add legend if there are multiple point tensors
    if point_tensors is not None and len(point_tensors) > 1:
        plt.legend()
    # Add colorbar with label
    cbar = plt.colorbar(im)

    plt.show()


def expand_2d_coordinates_to_nd(grid_2d, N, ind_0, ind_1, v=1.0):
    """
    Create tensor of shape (20,20,N) from grid coordinates.

    Parameters:
    -----------
    grid_2d : tensor of shape (20,20,2) containing coordinates
    N : int, dimension of the output space
    ind_0, ind_1 : int, indices where to put original coordinates
    v : float, value to fill other dimensions

    Returns:
    --------
    tensor of shape (20,20,N)
    """
    # Check indices
    if ind_0 == ind_1:
        raise ValueError("Indices must be different")
    if max(ind_0, ind_1) >= N or min(ind_0, ind_1) < 0:
        raise ValueError(f"Indices must be between 0 and {N - 1}")

    x_shape = grid_2d.shape[0]
    y_shape = grid_2d.shape[1]
    # Create base tensor filled with v
    result = tf.ones((x_shape, y_shape, N)) * v

    # Place coordinates at specified indices
    result = tf.tensor_scatter_nd_update(
        result,
        tf.constant([[i, j, ind_0] for i in range(x_shape) for j in range(y_shape)]),
        tf.reshape(grid_2d[..., 0], [-1])
    )
    result = tf.tensor_scatter_nd_update(
        result,
        tf.constant([[i, j, ind_1] for i in range(x_shape) for j in range(y_shape)]),
        tf.reshape(grid_2d[..., 1], [-1])
    )

    return result

def create_nd_grid(start: object = 0, stop: object = 10, step: object = 0.5, n_dims=4) -> object:
    """
    Create n-dimensional grid.

    Parameters:
    -----------
    start : float
        Start value for each dimension
    stop : float
        Stop value for each dimension
    step : float
        Step size for each dimension
    n_dims : int
        Number of dimensions

    Returns:
    --------
    list of arrays
        List of n arrays, each representing coordinates for one dimension
    """
    if n_dims < 1:
        raise ValueError("Number of dimensions must be positive")


    # Create sequence once
    x = tf.range(start, stop, delta=step)
    x = tf.cast(x, dtype=tf.float32)
    # Create n-dimensional meshgrid
    return tf.meshgrid(*[x] * n_dims, indexing='ij')


def sample_from_dict(person_dict,window_size=648, considered_weights=[0,0.5,1,2], samples_per_label_per_person=10):
    """ Calculate and project the feature space from model layer outputs for each person and label.  """

    samples_dict = {person: {} for person in
                         person_dict.keys()}  # {person: {weight: [] for weight in persons_dict[person]} for person in persons_dict.keys()}

    for person, weight_dict in person_dict.items():
        weight_dict_part = {weight: weight_dict[weight] for weight in considered_weights if
                            weight in weight_dict}

        for label, records in weight_dict_part.items():
            # tf.print('used_recored', used_records)
            if len(records)>0:
                snc1_batch = []
                snc2_batch = []
                snc3_batch = []
                for _ in range(samples_per_label_per_person):
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
                samples_dict[person][label] = persons_input_data
    return samples_dict


"""
A Keras callback that generates and plots heatmaps of model predictions on a 2D grid.
This callback creates visualizations either at the end of training or periodically during testing.

The callback works by:
1. Creating a submodel from a specified mean layer to the output
2. Generating a 2D grid of points and expanding it to the model's input dimension
3. Making predictions on these grid points
4. Visualizing the predictions as a heatmap

Parameters:
-----------
trial_dir : str
    Directory where plots will be saved
layer_name : str, default='mean_layer'
    Name of the layer from which to create the submodel
ind_0 : int, default=0
    First index for coordinate embedding in the expanded dimensions
ind_1 : int, default=1
    Second index for coordinate embedding in the expanded dimensions
grid_x_min : float, default=-48
    Minimum x value for the grid
grid_x_max : float, default=48
    Maximum x value for the grid
grid_step : float, default=0.125
    Step size for grid points
phase : str, default='train'
    Phase of training ('train' or 'test') determining when to generate plots

Methods:
--------
on_epoch_begin(epoch, logs=None):
    Updates the current epoch counter
on_train_end(logs=None):
    Generates heatmap at the end of training
on_test_end(epoch, logs=None):
    Generates heatmap every 5 epochs during testing phase
plot_heatmap():
    Creates the heatmap visualization of model predictions
"""
class HeatmapMeanPlotCallback(keras.callbacks.Callback):

    def __init__(self, person_dict, trial_dir, layer_name='mean_layer',
                 ind_0=0, ind_1 = 1, grid_x_min=-48, grid_x_max=48, grid_step = 0.125, v = 0, add_samples=False, person='Leeor',
                 phase='train'):
        super(HeatmapMeanPlotCallback, self).__init__()

        self.trial_dir = trial_dir
        self.person = person
        self.person_dict = person_dict
        self.layer_name = layer_name
        self.ind_0 = ind_0
        self.ind_1 = ind_1
        self.v = v
        self.grid_x_min = grid_x_min
        self.grid_x_max = grid_x_max
        self.grid_step = grid_step
        self.add_samples = add_samples
        self.phase = phase

        self.current_epoch = 0

    def on_epoch_begin(self, epoch, logs=None):
            self.current_epoch = epoch

    def on_train_end(self, logs=None):
        self.plot_heatmap()

    def on_test_end(self,epoch, logs=None):
        if (self.phase == 'test' or self.phase == 'Test') and  self.current_epoch % 5  == 0:
            self.plot_heatmap()


    def plot_heatmap(self):
        # Get the layer that applies mean operation
        mean_layer = self.model.get_layer(self.layer_name)
        window_size = self.model.input[0].shape[1] #648
        # Create submodel
        submodel = keras.Model(
            inputs=mean_layer.output,
            outputs=self.model.output
        )

        # Create grid
        x = create_nd_grid(start=self.grid_x_min, stop=self.grid_x_max, step=self.grid_step, n_dims=2)
        grid_shape_0 = x[0].shape[0]
        grid_points = tf.stack(x, axis=-1)
        N = submodel.inputs[0].shape[1]  # 36
        grid_slice = expand_2d_coordinates_to_nd(grid_points, N, self.ind_0, self.ind_1, v=self.v)
        reshaped_grid_slice = tf.reshape(grid_slice, (-1, N))

        # get model predictions
        model_predictions = submodel(reshaped_grid_slice)

        pred_on_grid = tf.reshape(model_predictions, (grid_shape_0, grid_shape_0, 1))
        pred_on_grid = tf.squeeze(pred_on_grid, axis=-1)

        if self.add_samples:
            # # get a persons dict
            # persons_dict = get_weight_file_from_dir('/home/wld-algo-6/Data/Sorted',
            #                                         multiplier=1)  # ('/home/wld-algo-6/DataCollection/Data')
            # Create layer_submodel
            layer_submodel = keras.Model(
                inputs=self.model.input,
                outputs=mean_layer.output
            )
            sample_dict = sample_from_dict(self.person_dict, window_size=window_size, considered_weights=[0,0.5,1,2], samples_per_label_per_person=15)
            user = self.person
            user_0 = layer_submodel(sample_dict[user][0])
            user_05 = layer_submodel(sample_dict[user][0.5])
            user_1 = layer_submodel(sample_dict[user][1])
            user_2 = layer_submodel(sample_dict[user][2])

            user_0 = tf.gather(user_0, [self.ind_0, self.ind_1], axis=1)
            user_05 = tf.gather(user_05, [self.ind_0, self.ind_1], axis=1)
            user_1 = tf.gather(user_1, [self.ind_0, self.ind_1], axis=1)
            user_2 = tf.gather(user_2, [self.ind_0, self.ind_1], axis=1)

            # List of point tensors
            point_tensors = [user_0, user_05, user_1, user_2]

            # Optional: define specific colors if you want
            colors = ['red', 'pink', 'blue', 'green']  # or use None for automatic color generation

            # Plot everything
            plot_heat_tensor(pred_on_grid, x_min=self.grid_x_min, x_max=self.grid_x_max, step=self.grid_step, point_tensors=point_tensors,
                             colors=colors)
        else:
            # plot predictions
            plot_heat_tensor(pred_on_grid, plot_grid=False)


if __name__ == "__main__":
    # take the model
    model_snc_path ='/home/wld-algo-6/Production/WeightEstimation2K/logs/16-01-2025-13-34-57/trials/trial_0/initial_pre_trained_model.keras'
    model_snc_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/19-01-2025-15-10-19/trials/trial_0/initial_pre_trained_model.keras'
    custom_objects = {'SEMGScatteringTransform': SEMGScatteringTransform}
    model = keras.models.load_model(model_snc_path, custom_objects=custom_objects,
                                    compile=True,
                                    safe_mode=False)

    # Get all layers
    layers = model.layers

    # Print layer names and types
    for layer in layers:
        print(f"Layer: {layer.name}, Type: {type(layer).__name__}")

    # Get the layer that applies mean operation
    mean_layer = model.get_layer('mean_layer')  # Make sure this name matches your layer name

    # Create submodel
    submodel = keras.Model(
        inputs=mean_layer.output,
        outputs=model.output
    )

    # get a persons dict
    # persons_dict = get_weight_file_from_dir('/home/wld-algo-6/Data/Sorted',
    #                                         multiplier=1)  # ('/home/wld-algo-6/DataCollection/Data')
    # # get mean scattering
    # output_dict, std_dict = mean_scattering_snc(persons_dict, window_size=648, samples_per_weight_per_person=25,scattering_type='SEMG', undersampling=3)
    # proj_dict, _ = apply_projection_to_dict(output_dict, n_components=2, perplexity=10, random_state=42, proj='none',
    #                                            metric="euclidean")

    # Create grid
    x_min = -0.001
    x_max = 0.0016
    x = create_nd_grid(start=x_min, stop=x_max, step=0.0001, n_dims=2)
    grid_shape_0 =x[0].shape[0]
    grid_points = tf.stack(x, axis=-1)
    ind_0 = 10
    ind_1 = 11
    N = submodel.inputs[0].shape[1] #36
    grid_slice = expand_2d_coordinates_to_nd(grid_points, N, ind_0, ind_1, v=-2.0)
    reshaped_grid_slice = tf.reshape(grid_slice, (-1, N))

    # get model predictions
    model_predictions = submodel(reshaped_grid_slice)

    pred_on_grid = tf.reshape(model_predictions, (grid_shape_0, grid_shape_0, 1))
    pred_on_grid = tf.squeeze(pred_on_grid, axis=-1)

    # plot predictions
    # plot_heat_tensor(pred_on_grid, plot_grid=False)

    # get a persons dict
    persons_dict = get_weight_file_from_dir('/home/wld-algo-6/Data/Sorted',
                                            multiplier=1)  # ('/home/wld-algo-6/DataCollection/Data')
    # Create layer_submodel
    layer_submodel = keras.Model(
        inputs=model.input,
        outputs=mean_layer.output
    )
    sample_dict = sample_from_dict(persons_dict)
    user = 'Leeor'
    user_0 = layer_submodel(sample_dict[user][0])
    user_05 = layer_submodel(sample_dict[user][0.5])
    user_1 = layer_submodel(sample_dict[user][1])
    user_2 = layer_submodel(sample_dict[user][2])

    user_0 = tf.gather(user_0, [ind_0, ind_1], axis=1)
    user_05 = tf.gather(user_05, [ind_0, ind_1], axis=1)
    user_1 = tf.gather(user_1, [ind_0, ind_1], axis=1)
    user_2 = tf.gather(user_2, [ind_0, ind_1], axis=1)

    # List of point tensors
    point_tensors = [user_0, user_05, user_1, user_2]

    # Optional: define specific colors if you want
    colors = ['red', 'pink','blue', 'green']  # or use None for automatic color generation

    # Plot everything
    plot_heat_tensor(pred_on_grid,x_min=x_min, x_max=x_max, step=0.0001, point_tensors=point_tensors, colors=colors)
    ttt = 1


