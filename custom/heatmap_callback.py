import keras
from custom.layers import SEMGScatteringTransform
import tensorflow as tf
# from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_heat_tensor(tensor, title = 'Heatmap', colomn_index = 'Column Index', row_index = 'Row Index', plot_grid=True):
    # x_values = np.arange(-48, 48, 0.5)  # example x coordinates
    # y_values = np.arange(-48, 48, 0.5)
    # Convert tensor to numpy if needed
    if tf.is_tensor(tensor):
        data = tensor.numpy()
    else:
        data = tensor

    plt.figure(figsize=(10, 8))
    sns.heatmap(data, cmap='viridis',#xticklabels=x_values,  # Custom x-axis labels
                annot=False)  # annot=True shows values in cells
    plt.title(title)
    plt.xlabel(colomn_index)
    plt.ylabel(row_index)
    # Add grid
    if plot_grid:
        plt.grid(True)
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

    def __init__(self, trial_dir, layer_name='mean_layer',
                 ind_0=0, ind_1 = 1, grid_x_min=-48, grid_x_max=48, grid_step = 0.125, v = 0,
                 phase='train'):
        super(HeatmapMeanPlotCallback, self).__init__()

        self.trial_dir = trial_dir
        self.layer_name = layer_name
        self.ind_0 = ind_0
        self.ind_1 = ind_1
        self.v = v
        self.grid_x_min = grid_x_min
        self.grid_x_max = grid_x_max
        self.grid_step = grid_step
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

        # plot predictions
        plot_heat_tensor(pred_on_grid, plot_grid=False)


if __name__ == "__main__":
    # take the model
    model_snc_path ='/home/wld-algo-6/Production/WeightEstimation2K/logs/16-01-2025-13-34-57/trials/trial_0/initial_pre_trained_model.keras'
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
    x = create_nd_grid(start=-48, stop=48, step=0.125, n_dims=2)
    grid_shape_0 =x[0].shape[0]
    grid_points = tf.stack(x, axis=-1)
    ind_0 = 0
    ind_1 = 1
    N = submodel.inputs[0].shape[1] #36
    grid_slice = expand_2d_coordinates_to_nd(grid_points, N, ind_0, ind_1, v=-2.0)
    reshaped_grid_slice = tf.reshape(grid_slice, (-1, N))

    # get model predictions
    model_predictions = submodel(reshaped_grid_slice)

    pred_on_grid = tf.reshape(model_predictions, (grid_shape_0, grid_shape_0, 1))
    pred_on_grid = tf.squeeze(pred_on_grid, axis=-1)

    # plot predictions
    plot_heat_tensor(pred_on_grid, plot_grid=False)

    ttt = 1


