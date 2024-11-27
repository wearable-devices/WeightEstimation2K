
from utils.get_data import get_weight_file_from_dir
from pathlib import Path
from datetime import datetime
import keras

def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    # Define the search space and sample parameter values
    snc_window_size_hp = trial.suggest_int("snc_window_size", 162, 1800, step=18)  # 1044#
    addition_weight_hp = 0#trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)
    epoch_num =  40
    epoch_len = 5  # None
    use_pretrained_model = True  # trial.suggest_categorical('use_pretrained_model',[True

def logging_dirs():
    package_directory = Path(__file__).parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)
    trials_dir = log_dir / "trials"
    trials_dir.mkdir(exist_ok=True)

    return logs_root_dir, log_dir, trials_dir


if __name__ == "__main__":
    persons_for_train_initial_model = ['Avihoo', 'Aviner', 'Shai', #'HishamCleaned',
                                       'Alisa','Molham']
    persons_for_test = [ 'Leeor',
                        'Liav',
                         'Daniel',
                         'Foad',
                        'Asher2', 'Lee',
                   'Ofek',
       'Tom', #'Guy'
                        ]
    persons_for_plot = persons_for_test

    # USE_PRETRAINED_MODEL=True
    file_dir = '/home/wld-algo-6/DataCollection/Data'
    person_dict = get_weight_file_from_dir(file_dir)

    logs_root_dir, log_dir, trials_dir = logging_dirs()

    hint_model_path = '/home/wld-algo-6/Production/WeightEstimation2K/UserIDmodels/model_trial_2 (1).keras'