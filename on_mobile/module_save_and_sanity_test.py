import tensorflow as tf
import os
from on_mobile.on_device_module import OnDeviceModel
# from training_playground.models import create_attention_weight_estimation_model
# from training_playground.constants import *
from pathlib import Path
from datetime import datetime
import numpy as np
from db_generators.get_db import process_file
import keras
from custom.layers import SEMGScatteringTransform

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
    custom_objects = {
        'SEMGScatteringTransform': SEMGScatteringTransform,
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


if __name__ == "__main__":
    logs_root_dir, log_dir = logging_dirs()

    # SAVED_MODEL_DIR = "saved_model"
    SAVED_MODEL_DIR = str(log_dir)
    original_model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/24-03-2025-09-54-54/trials/trial_5/initial_pre_trained_model.keras'

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
    window_size=648

    mobile_friendly_model=create_mobile_friendly_model(original_model_path,(window_size,))
    mobile_friendly_model.save(os.path.join(log_dir,
                                 'mobile_friendly_model' + '.keras'),
                    save_format='keras')
    model_path = os.path.join(log_dir, 'mobile_friendly_model.keras')

    m = OnDeviceModel(model_path)
    tf.saved_model.save(
        m,
        SAVED_MODEL_DIR,
        signatures={
            'train':
                m.train.get_concrete_function(),
            'infer':
                m.infer.get_concrete_function(),
            'save':
                m.save.get_concrete_function(),
            'restore':
                m.restore.get_concrete_function(),
        })

    # Convert the model
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS  # enable TensorFlow ops.
    ]

    #NEW
    converter.allow_custom_ops = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    tflite_model_path = os.path.join(log_dir, 'snc_weight_estimation_model.tflite')
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)

    print('finished saving module to tflite')

    # Test
    SNC_WINDOW_SIZE=648 # or take from the model
    file_path = '/home/wld-algo-6/Data/Sorted/Leeor/weight_estimation/Train/Leeor_1_weight_0_0_Leaning_M.csv'
    snc1_data, snc2_data, snc3_data = process_file(file_path)

    valid_start_range = max(0, len(snc1_data) - SNC_WINDOW_SIZE + 1)

    start = np.random.randint(0, valid_start_range)
    snc_1_window = np.expand_dims(snc1_data[start:start + SNC_WINDOW_SIZE], axis=0)
    snc_2_window = np.expand_dims(snc2_data[start:start + SNC_WINDOW_SIZE], axis=0)
    snc_3_window = np.expand_dims(snc3_data[start:start + SNC_WINDOW_SIZE], axis=0)

    label = np.array([[0]])

    save_path = os.path.join(log_dir, 'weights_before')
    m.save(save_path)

    for _ in range(100):
        m.train(snc1=snc_1_window, snc2=snc_2_window, snc3=snc_3_window, y=label)#[label, label, label, label])

    save_path = os.path.join(log_dir, 'weights_after')
    m.save(save_path)

    m.restore(save_path)
