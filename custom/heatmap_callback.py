import keras
from custom.layers import SEMGScatteringTransform
import tensorflow as tf
from sklearn.decomposition import PCA


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
    # Create n-dimensional meshgrid
    return tf.meshgrid(*[x] * n_dims, indexing='ij')


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

    x1 = tf.range(start=0, limit=10, delta=0.5, dtype=None, name='range')
    x2 = tf.range(start=0, limit=10, delta=0.5, dtype=None, name='range')
    x3 = tf.range(start=0, limit=10, delta=0.5, dtype=None, name='range')
    x4 = tf.range(start=0, limit=10, delta=0.5, dtype=None, name='range')
    X,Y, Z, T = tf.meshgrid(x1,x2, x3, x4)

    print('x', X.shape)

    x = create_nd_grid(start=0, stop=3, step=1.0, n_dims=2)
    grid_points = tf.stack(x, axis=-1)
    # print(grid_points)
    ind_0 = 0
    ind_1 = 1

    N = submodel.inputs[0].shape[1] #36

    grid_slice = expand_2d_coordinates_to_nd(grid_points, N, ind_0, ind_1, v=0.0)
    # print(grid_slice)
    ttt = 1


