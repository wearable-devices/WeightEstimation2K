import tensorflow as tf
import os
from on_mobile.on_device_training import OnDeviceModel
from training_playground.models import create_attention_weight_estimation_model
from training_playground.constants import *
from pathlib import Path
from datetime import datetime
import numpy as np
from db_generators.get_db import process_file

def logging_dirs():
    package_directory = Path(__file__).parent.parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir

if __name__ == "__main__":
    ogs_root_dir, log_dir = logging_dirs()

    SAVED_MODEL_DIR = "saved_model"
    m = OnDeviceModel()
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
    converter.experimental_enable_resource_variables = True
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    tflite_model_path = os.path.join(log_dir, 'snc_weight_estimation_model.tflite')
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)


    file_path = '/home/wld-algo-5/Data/Data_2023/AllCSVData/Train_edited/AvihooKeret/DragNDropSwipeFling/dragNdrop_1_cleaned.csv'
    snc1_data, snc2_data, snc3_data = process_file(file_path)

    valid_start_range = max(0, len(snc1_data) - SNC_WINDOW_SIZE + 1)

    start = np.random.randint(0, valid_start_range)
    snc_1_window = np.expand_dims(snc1_data[start:start + SNC_WINDOW_SIZE], axis=0)
    snc_2_window = np.expand_dims(snc2_data[start:start + SNC_WINDOW_SIZE], axis=0)
    snc_3_window = np.expand_dims(snc3_data[start:start + SNC_WINDOW_SIZE], axis=0)

    label = np.array([[1]])

    save_path = os.path.join(log_dir, 'weights_before')
    m.save(save_path)

    for _ in range(100):
        m.train(snc1=snc_1_window, snc2=snc_2_window, snc3=snc_3_window, y=[label, label, label, label])

    save_path = os.path.join(log_dir, 'weights_after')
    m.save(save_path)

    # m.restore(save_path)
