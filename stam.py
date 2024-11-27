
from keras.callbacks import TensorBoard
import optuna
import os

from optuna_dashboard import save_plotly_graph_object, save_note
from datetime import datetime
from optuna.trial import TrialState
from models import *
from pathlib import Path
from db_generators.generators import create_data_for_userId_model,UserIdModelGenerator, create_data_for_model
from utils.get_data import get_weight_file_from_dir
from constants import *
from custom.callbacks import *
# from db_generators.create_person_dict import *
from tensorflow.keras.callbacks import ModelCheckpoint
from custom.callbacks import get_layer_output
from custom.for_debug import WeightMonitorCallback

from db_generators.generators import preprocess_person_data
from models_dir.model_fusion import one_sensor_model_fusion
def logging_dirs():
    package_directory = Path(__file__).parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir


snc_window_size = 1062

logs_root_dir, log_dir = logging_dirs()
file_dir = r"C:\Users\sofia.a\PycharmProjects\DATA_2024\Sorted_old"#'/home/wld-algo-6/DataCollection/Data'
person_dict = get_weight_file_from_dir(file_dir)
person_zero_dict = {person_name: weight_dict[0] for person_name, weight_dict in person_dict.items() if 0 in weight_dict.keys()}

persons = [#'Alisa', 'Asher2', 'Avihoo', 'Aviner', 'HishamCleaned','Lee',
               'Leeor',
               'Daniel',
              # 'Liav',
              #  'Foad',
              # 'Molham',
              # 'Ofek'
               ]#,'Guy']
# train_data, test_data, person_to_idx = preprocess_person_data(person_zero_dict,
#                                                                   window_size_snc=snc_window_size,
#                                                                   window_step=54,
#                                                                   persons=persons,
#                                                                   print_stat=False,
#                                                                   multiplier=10)

# After preprocessing
# if check_for_nans(train_data) or check_for_nans(test_data):
#     raise ValueError("NaN or Inf values found in the preprocessed data")

# Create person to index mapping
person_to_idx = {name: idx for idx, name in enumerate(persons)}
num_persons = len(person_to_idx) if persons == 'all' else len(persons)

model_snc_path = r"C:\Users\sofia.a\PycharmProjects\Production\WeightEstimation2K\MODELS\initial_pre_trained_model.keras"
model_1 = keras.models.load_model(model_snc_path, #custom_objects=custom_objects,
                                               #compile=True,
                                               safe_mode=False)
model = one_sensor_model_fusion(model_1, model_1, model_1,
                             fusion_type='majority_vote',
                             window_size_snc=snc_window_size,
                             trainable=False,
                             optimizer='Adam', learning_rate=0.0016,compile=True,
                             )
epoch_len = 10#None
batch_size = 110
labels_to_balance = [0,0.5,1,2]
train_ds = create_data_for_model(person_dict, snc_window_size, batch_size, labels_to_balance, epoch_len, persons,
                          data_mode='Train', contacts=['M'])
val_ds = create_data_for_model(person_dict, snc_window_size, batch_size, labels_to_balance, epoch_len, persons,
                          data_mode='Test', contacts=['M'])


model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=[NanCallback()],
        epochs=10,
        batch_size=128
    )
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
