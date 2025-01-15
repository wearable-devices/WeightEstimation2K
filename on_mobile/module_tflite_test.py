import tensorflow as tf
import os
# from on_mobile.on_device_training import OnDeviceModel
# from training_playground.models import create_attention_weight_estimation_model
# from training_playground.constants import *
from pathlib import Path
from datetime import datetime
import numpy as np
from db_generators.get_db import process_file
import keras
from custom.layers import SEMGScatteringTransform

SNC_WINDOW_SIZE = 648

def logging_dirs():
    package_directory = Path(__file__).parent.parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir

if __name__ == "__main__":
    model_path= '/home/wld-algo-6/Production/WeightEstimation2K/logs/15-01-2025-14-08-35/snc_weight_estimation_model.tflite'#'#os.path.join('/home/wld-algo-5/Production/WeightEstimation/logs/11-09-2024-15-01-35', 'snc_weight_estimation_model.tflite')

    interpreter = tf.lite.Interpreter(model_path=model_path)
    # Get signature list
    signatures = interpreter.get_signature_list()
    # Choose a specific signature - Restore
    signature_name = "restore"
    signature_runner = interpreter.get_signature_runner(signature_name)
    # The shape and type should match what's expected by the model
    weights_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/15-01-2025-14-08-35/weights_before'
    input_tensor = np.array([weights_path], dtype=np.string_)

    # Run the restore function
    output = signature_runner(checkpoint_path=input_tensor)
    #output = signature_runner(input_data='/home/wld-algo-5/Production/WeightEstimation/logs/11-09-2024-14-36-17/weights_after')
    # Run the specific function
    # m.restore(save_path)

    # Choose a specific signature - Infer
    signature_name = "infer"
    signature_runner = interpreter.get_signature_runner(signature_name)
    # The shape and type should match what's expected by the model
    weights_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/15-01-2025-14-08-35/weights_after'
    # input_tensor = #np.array([weights_path], dtype=np.string_)

    # Run the restore function
    file_path = '/home/wld-algo-6/Data/Sorted/Leeor/weight_estimation/Train/Leeor_1_weight_0_0_Leaning_M.csv'
    snc1_data, snc2_data, snc3_data = process_file(file_path)

    valid_start_range = max(0, len(snc1_data) - SNC_WINDOW_SIZE + 1)

    start = np.random.randint(0, valid_start_range)
    snc_1_window = np.array(np.expand_dims(snc1_data[start:start + SNC_WINDOW_SIZE], axis=0), dtype=np.float32)
    snc_2_window = np.array(np.expand_dims(snc2_data[start:start + SNC_WINDOW_SIZE], axis=0), dtype=np.float32)
    snc_3_window = np.array(np.expand_dims(snc3_data[start:start + SNC_WINDOW_SIZE], axis=0), dtype=np.float32)
    tflite_output = signature_runner(snc_1=snc_1_window,snc_2=snc_2_window, snc_3=snc_3_window)
    print('tflite_output',tflite_output)

    keras_mode_path ='/home/wld-algo-6/Production/WeightEstimation2K/logs/15-01-2025-11-35-38/trials/trial_0/initial_pre_trained_model.keras'
    custom_objects = {
        'SEMGScatteringTransform': SEMGScatteringTransform,
    }
    keras_model = keras.models.load_model(
        keras_mode_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    keras_output = keras_model([snc_1_window, snc_2_window, snc_3_window])
    print('keras_output',keras_output)


    # Do the same in keras (run the model on the same windows). Compare model predictions
