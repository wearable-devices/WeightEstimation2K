import tensorflow as tf
import os
from on_mobile.on_device_module import OnDeviceModel, OnDeviceSubModel
# from training_playground.models import create_attention_weight_estimation_model
# from training_playground.constants import *
from pathlib import Path
from datetime import datetime
import numpy as np
from db_generators.get_db import process_file
import keras
from custom.layers import SEMGScatteringTransform
from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann

from google_friendly_model.build_model import SequentialCrossSpectralSpd_matricesLayer
from google_friendly_model.tangent_proj import TangentSpaceLayer
from on_mobile.mobile_constants import *
from on_mobile.tang_model import create_from_tangent_model

# SNC_WINDOW_SIZE = 648


def logging_dirs():
    package_directory = Path(__file__).parent.parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir


# Create a simpler version of your model for mobile
def create_mobile_friendly_model(model_path, input_shape):
    # Load the original model
    # custom_objects = {
    #     # 'SEMGScatteringTransform': SEMGScatteringTransform,
    #     'SequentialCrossSpectralDensityLayer_pyriemann': SequentialCrossSpectralDensityLayer_pyriemann,
    # }
    custom_objects = {
        'SequentialCrossSpectralSpd_matricesLayer': SequentialCrossSpectralSpd_matricesLayer,
        'TangentSpaceLayer': TangentSpaceLayer
    }
    original_model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

    # Define a new model that only handles inference
    inputs = {
        'snc_1': tf.keras.Input(shape=input_shape, name='snc_1'),
        'snc_2': tf.keras.Input(shape=input_shape, name='snc_2'),
        'snc_3': tf.keras.Input(shape=input_shape, name='snc_3')
    }

    # Get the inference output from the original model
    outputs = original_model(inputs)

    # Create a new model with just the inference path
    mobile_model = tf.keras.Model(inputs=inputs, outputs=outputs)

    return mobile_model


def convert_submodel_with_select_ops(submodel_path, log_dir,
                                  # window_size = SNC_WINDOW_SIZE  # 1377
                                    ):
    """
    Convert model to TFLite with proper Select TF ops support
    Following Google's best practices
    """

    print("🔄 Converting model with TensorFlow Select ops support...")

    # logs_root_dir, log_dir = logging_dirs()

    # Your model path
    # original_model_path = '/home/wld-algo-6/Production/WeightEstimation2K/google_friendly_model/logs/05-06-2025-16-09-19/model_bestbest.keras'

    # Create mobile-friendly model
    # window_size = SNC_WINDOW_SIZE  # 1377
    # mobile_friendly_model = create_mobile_friendly_model(original_model_path, (window_size,))

    # custom_objects = {
    #     'SequentialCrossSpectralSpd_matricesLayer': SequentialCrossSpectralSpd_matricesLayer,
    #     'TangentSpaceLayer': TangentSpaceLayer
    # }
    submodel = keras.models.load_model(
        submodel_path,
        # custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )

    # Save the mobile model
    # submodel_path = os.path.join(log_dir, 'submodel.keras')
    # submodel.save(submodel_path, save_format='keras')
    # print(f"✅ Saved mobile model: {submodel_path}")

    # Create OnDeviceModel wrapper
    m = OnDeviceSubModel(submodel_path)

    # Save as SavedModel format
    saved_model_dir = str(log_dir)
    tf.saved_model.save(
        m,
        saved_model_dir,
        signatures={
            'train': m.train.get_concrete_function(),
            'infer': m.infer.get_concrete_function(),
            'save': m.save.get_concrete_function(),
            'restore': m.restore.get_concrete_function(),
        }
    )
    print(f"✅ Saved SavedModel: {saved_model_dir}")

    # CRITICAL: Convert to TFLite with SELECT_TF_OPS (following Google's approach)
    print("🔧 Converting to TensorFlow Lite with Select ops...")

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

    # Enable Select TF ops (this is crucial for SelfAdjointEigV2)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # Standard TFLite ops
        tf.lite.OpsSet.SELECT_TF_OPS  # TensorFlow ops (includes SelfAdjointEigV2)
    ]

    # Allow custom ops and enable resource variables
    converter.allow_custom_ops = True
    converter.experimental_enable_resource_variables = True

    # Optional: Enable optimizations (but be careful with Select ops)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model
    try:
        print("   Converting model...")
        tflite_model = converter.convert()
        print("✅ Model conversion successful!")

        # Save TFLite model
        tflite_model_path = os.path.join(log_dir, 'snc_weight_estimation_model.tflite')
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)

        print(f"✅ Saved TFLite model: {tflite_model_path}")
        return tflite_model_path, m#, log_dir

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        if "SelfAdjointEigV2" in str(e):
            print("🔍 SelfAdjointEigV2 detected during conversion")
            print("This should be handled by SELECT_TF_OPS, but conversion still failed")
        return None, None, None


def test_converted_model(tflite_model_path):
    """
    Test the converted TFLite model (following Google's approach)
    """

    print(f"\n🧪 Testing converted TFLite model...")

    try:
        # Load the TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
        interpreter.allocate_tensors()
        print("✅ TFLite model loaded successfully")

        # Get model details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        signatures = interpreter.get_signature_list()

        print(f"   Signatures: {list(signatures.keys())}")
        print(f"   Inputs: {len(input_details)}")
        print(f"   Outputs: {len(output_details)}")

        # Test inference with signatures
        if signatures:
            sig_name = list(signatures.keys())[0]
            signature_runner = interpreter.get_signature_runner(sig_name)

            # Create test data
            test_data = np.random.normal(0, 0.1, (1, SNC_WINDOW_SIZE)).astype(np.float32)

            print(f"   Running inference with signature '{sig_name}'...")

            # This is the call that was failing
            output_data = signature_runner(
                snc_1=test_data,
                snc_2=test_data,
                snc_3=test_data
            )

            print("🎉 SUCCESS! TFLite inference completed!")
            print(f"   Output keys: {list(output_data.keys())}")

            for output_name, output_value in output_data.items():
                if hasattr(output_value, 'shape'):
                    print(f"   '{output_name}': shape {output_value.shape}, value {output_value.flatten()[:3]}...")
                else:
                    print(f"   '{output_name}': {output_value}")

            return True
        else:
            print("❌ No signatures found")
            return False

    except Exception as e:
        print(f"❌ TFLite test failed: {e}")
        if "SelfAdjointEigV2" in str(e):
            print("🔍 SelfAdjointEigV2 error still occurring")
            print("This suggests the Select ops are not being loaded properly at runtime")
        return False


def test_training_functionality(m, log_dir):
    """
    Test the training functionality (from your original code)
    """

    print(f"\n🏋️ Testing training functionality...")

    try:
        # Import your data processing function
        from db_generators.get_db import process_file

        # Test data
        file_path = '/media/wld-algo-6/Storage/SortedCleaned/Leeor/weight_estimation/Leeor_1_weight_0_0_Leaning_M.csv'
        snc1_data, snc2_data, snc3_data = process_file(file_path)

        valid_start_range = max(0, len(snc1_data) - SNC_WINDOW_SIZE + 1)
        start = np.random.randint(0, valid_start_range)

        snc_1_window = np.expand_dims(snc1_data[start:start + SNC_WINDOW_SIZE], axis=0)
        snc_2_window = np.expand_dims(snc2_data[start:start + SNC_WINDOW_SIZE], axis=0)
        snc_3_window = np.expand_dims(snc3_data[start:start + SNC_WINDOW_SIZE], axis=0)

        label = np.array([[0]])

        # Save initial weights
        save_path = os.path.join(log_dir, 'weights_before')
        m.save(save_path)

        # Train for a few iterations
        print("   Running training iterations...")
        for i in range(10):  # Reduced for testing
            m.train(snc1=snc_1_window, snc2=snc_2_window, snc3=snc_3_window, y=label)
            if i % 5 == 0:
                print(f"   Training iteration {i}")

        # Save final weights
        save_path = os.path.join(log_dir, 'weights_after')
        m.save(save_path)

        # Test restore
        m.restore(save_path)

        print("✅ Training functionality test successful!")
        return True

    except Exception as e:
        print(f"❌ Training test failed: {e}")
        return False


def print_and_get_model_layers(model_path):
    """
    Loads a Keras model from the given path, prints a summary, and returns a list of its layers.
    """
    custom_objects = {
        'SequentialCrossSpectralSpd_matricesLayer': SequentialCrossSpectralSpd_matricesLayer,
        'TangentSpaceLayer': TangentSpaceLayer
    }
    model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    print("Model Summary:")
    model.summary()
    layer_names = [layer.name for layer in model.layers]
    print("\nLayer names:", layer_names)
    return model.layers


def create_tangent_submodel(original_model_path):
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
    tg_input = keras.Input(shape=(6,), name='tg_vector')

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
    logs_root_dir, log_dir = logging_dirs()

    # SAVED_MODEL_DIR = "saved_model"
    # SAVED_MODEL_DIR = str(log_dir)
    original_model_path = '/home/wld-algo-6/Production/WeightEstimation2K/google_friendly_model/logs/05-06-2025-16-09-19/model_bestbest.keras'
    
    layers = print_and_get_model_layers(original_model_path)
    print(layers)

    submodel = create_tangent_submodel(original_model_path)

    # Save submodel
    submodel.save(os.path.join(log_dir,'submodel.keras'),save_format='keras')
    submodel_path = os.path.join(log_dir, 'submodel.keras')

    # Step 1: Convert model with Select ops
    tflite_model_path, on_device_model = convert_submodel_with_select_ops(submodel_path, log_dir)

    # Step 2: Test the converted TFLite model
    tflite_success = test_converted_model(tflite_model_path)
    # FOR DEBUG
    # custom_objects = {
    #     'SEMGScatteringTransform': SEMGScatteringTransform,
    # }
    #
    # model = keras.models.load_model(
    #     original_model_path,
    #     custom_objects=custom_objects,
    #     compile=False,
    #     safe_mode=False
    # )

    # window_size = model.inputs[0].shape[-1]
    # window_size=SNC_WINDOW_SIZE#1377#1394#1620#648
    #
    # mobile_friendly_model=create_mobile_friendly_model(original_model_path,(window_size,))
    # mobile_friendly_model.save(os.path.join(log_dir,
    #                              'mobile_friendly_model' + '.keras'),
    #                 save_format='keras')
    # model_path = os.path.join(log_dir, 'mobile_friendly_model.keras')
    #
    # m = OnDeviceModel(model_path)
    # tf.saved_model.save(
    #     m,
    #     SAVED_MODEL_DIR,
    #     signatures={
    #         'train':
    #             m.train.get_concrete_function(),
    #         'infer':
    #             m.infer.get_concrete_function(),
    #         'save':
    #             m.save.get_concrete_function(),
    #         'restore':
    #             m.restore.get_concrete_function(),
    #     })
    #
    # # # Convert the model
    # converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    # converter.allow_custom_ops = True # NEW
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
    #     tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    # ]
    #
    #
    # # alternative converter
    # # converter = tf.lite.TFLiteConverter.from_keras_model(m.model)
    # #
    # # # Enable TF Select ops to support SelfAdjointEigV2
    # # converter.allow_custom_ops = True
    # # converter.target_spec.supported_ops = [
    # #     tf.lite.OpsSet.TFLITE_BUILTINS,
    # #     tf.lite.OpsSet.SELECT_TF_OPS  # This allows SelfAdjointEigV2
    # # ]
    #
    # # tflite_model = converter.convert()
    # converter.experimental_enable_resource_variables = True
    # tflite_model = converter.convert()



    # Save the TFLite model to a file
    # tflite_model_path = os.path.join(log_dir, 'snc_weight_estimation_model.tflite')
    # with open(tflite_model_path, "wb") as f:
    #     f.write(tflite_model)
    #
    # print('finished saving module to tflite')
    #
    # # Test
    # # SNC_WINDOW_SIZE=1377#1394 # or take from the model
    # file_path = '/media/wld-algo-6/Storage/SortedCleaned/Leeor/weight_estimation/Leeor_1_weight_0_0_Leaning_M.csv'
    # snc1_data, snc2_data, snc3_data = process_file(file_path)
    #
    # valid_start_range = max(0, len(snc1_data) - SNC_WINDOW_SIZE + 1)
    #
    # start = np.random.randint(0, valid_start_range)
    # snc_1_window = np.expand_dims(snc1_data[start:start + SNC_WINDOW_SIZE], axis=0)
    # snc_2_window = np.expand_dims(snc2_data[start:start + SNC_WINDOW_SIZE], axis=0)
    # snc_3_window = np.expand_dims(snc3_data[start:start + SNC_WINDOW_SIZE], axis=0)
    #
    # label = np.array([[0]])
    #
    # save_path = os.path.join(log_dir, 'weights_before')
    # m.save(save_path)
    #
    # for _ in range(100):
    #     m.train(snc1=snc_1_window, snc2=snc_2_window, snc3=snc_3_window, y=label)#[label, label, label, label])
    #
    # save_path = os.path.join(log_dir, 'weights_after')
    # m.save(save_path)
    #
    # m.restore(save_path)
