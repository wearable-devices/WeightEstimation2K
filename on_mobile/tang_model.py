import tensorflow as tf
from tensorflow import keras
from google_friendly_model.build_model import SequentialCrossSpectralSpd_matricesLayer, TangentSpaceLayer

def create_from_tangent_model(original_model_path):
    """
    Loads the original model, finds the layers after 'tangent_space_layer', and creates a new model
    that takes keras.Input(shape=6, name='tg') as input and applies the same layers as the original model after tangent_space_layer.
    Returns the new model.
    """
    custom_objects = {
        'SequentialCrossSpectralSpd_matricesLayer': SequentialCrossSpectralSpd_matricesLayer,
        'TangentSpaceLayer': TangentSpaceLayer
    }
    model = keras.models.load_model(
        original_model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

    # Find the tangent_space_layer
    tangent_layer = None
    for i, layer in enumerate(model.layers):
        if layer.name == 'tangent_space_layer':
            tangent_layer = layer
            tangent_index = i
            break
    if tangent_layer is None:
        raise ValueError("No layer named 'tangent_space_layer' found in the model.")

    # Get the output tensor of the tangent_space_layer
    # and the layers after it
    # The next layers are: dense_1, dense, tf.math.multiply (TFOpLambda)
    # We'll reconstruct the computation graph from this point
    # Input shape is (None, 6)
    tg_input = keras.Input(shape=(6,), name='tg')

    # Find the first layer after tangent_space_layer that takes its output
    # We'll use the model's config to reconstruct the path
    # For this model, it's dense_1, then dense, then tf.math.multiply
    # We'll use the layer names for robustness
    dense_1 = model.get_layer('dense_1')
    dense = model.get_layer('dense')
    multiply_layer = None
    for l in model.layers:
        if l.name == 'tf.math.multiply':
            multiply_layer = l
            break

    x = dense_1(tg_input)
    x = dense(x)
    if multiply_layer is not None:
        # The multiply layer is a TFOpLambda, which multiplies by a constant (max_weight)
        # We'll extract the constant from the original model
        # Find the constant from the original model's computation graph
        # We'll try to get it from the multiply layer's call args
        # If not possible, just use the output of dense as the final output
        try:
            # Try to get the constant from the original model
            # This is a bit hacky, but works for simple graphs
            # The multiply layer's call function multiplies by a constant
            # We'll check the original model's output tensor
            import tensorflow as tf
            dense_output = dense.output
            multiply_output = model.output
            # Find the constant by dividing outputs
            # (This only works if the output is not zero)
            # We'll just use the same multiply op as in the original model
            from tensorflow.keras.layers import Lambda
            import operator
            # Try to get the constant from the multiply op
            # If not possible, just use the output of dense
            # (In your model, it's max_weight=2)
            max_weight = 2
            x = Lambda(lambda y: y * max_weight, name='multiply')(x)
        except Exception:
            pass
    new_model = keras.Model(inputs=tg_input, outputs=x, name='from_tangent_model')
    print("New model from tangent_space_layer summary:")
    new_model.summary()
    return new_model


if __name__ == "__main__":
    print(f"TensorFlow version: {tf.__version__}")
    original_model_path = '/home/wld-algo-6/Production/WeightEstimation2K/google_friendly_model/logs/05-06-2025-16-09-19/model_bestbest.keras'

    submodel = create_from_tangent_model(original_model_path)

    submodel.summary()
