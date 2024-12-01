
from utils.get_data import get_weight_file_from_dir
from pathlib import Path
from datetime import datetime
import keras
from models import one_sensor_weight_estimation_with_zeroidhint_model
from db_generators.generators import create_data_for_model
import  os
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from custom.callbacks import *
import optuna
from optuna.trial import TrialState
from optuna_dashboard import save_plotly_graph_object

def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    # Define the search space and sample parameter values
    snc_window_size_hp = 1548#trial.suggest_int("snc_window_size", 162, 1800, step=18)  # 1044#
    addition_weight_hp = 0#trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)
    epoch_num =  40
    epoch_len = 5  # None
    use_pretrained_model = True  # trial.suggest_categorical('use_pretrained_model',[True
    parameters_dict = {'sensor_num':2,
                       'window_size_snc': snc_window_size_hp}
    hint_model = keras.models.load_model(hint_model_path, #custom_objects=custom_objects,
                                         compile=False,
                                         safe_mode=False)
    model = one_sensor_weight_estimation_with_zeroidhint_model(hint_model = hint_model,**parameters_dict)
    model.summary()

    # pruning_callback = optuna.integration.tensorboard.TensorBoardCallback(trial)
    trial_dir = trials_dir / f"trial_{trial.number}"  # Specific trial
    trial_dir.mkdir(exist_ok=True)
    trial_dir = str(trial_dir)
    trial.set_user_attr("directory", trial_dir)  # Attribute can be seen in optuna-dashboard
    model_name = f"model_trial_{trial.number}"

    if use_pretrained_model:
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, BATCH_SIZE, labels_to_balance, epoch_len,
                                         used_persons=persons_for_train_initial_model, data_mode='Train', contacts=['M'])
        val_ds = train_ds

        # Before training, verify model configuration
        print("Model configuration:")
        print(model.get_config())

        model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            callbacks=[ModelCheckpoint(
                filepath=os.path.join(trial_dir,
                                      f"pre_trained_model_trial_{trial.number}__epoch_{{epoch:03d}}.weights.h5"),
                # Added .weights.h5
                verbose=1,
                save_weights_only=True,
                save_freq='epoch'),
                TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
            ],
            epochs=20,
            validation_data=val_ds,
            verbose=1,
        )



    initial_model_path = os.path.join(trial_dir, 'initial_pre_trained_model' + '.keras')
    model.save(initial_model_path, save_format='keras')
    print(f'Model saved to {trial_dir}')
    results_for_same_parameters = []
    # for _ in range(1):
    personal_metrics_dict = {}
    for person in persons_for_test:
        print(f'Training on {person}')
        # attention_snc_model_parameters_dict['max_weight'] = 2.5#max(train_dict[person].keys()) + 0.5
        if use_pretrained_model:
            model_snc_path = initial_model_path
            # custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain}
            model = keras.models.load_model(model_snc_path,# custom_objects=custom_objects,
                                               compile=True,
                                               safe_mode=False)
        else:
            model = one_sensor_weight_estimation_with_zeroidhint_model(**parameters_dict)

        # total_params, trainable_params, non_trainable_params = count_parameters(model)
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, BATCH_SIZE, labels_to_balance, epoch_len,
                                          [person], data_mode='Train', contacts=['M'])
        val_ds = create_data_for_model(person_dict, snc_window_size_hp, BATCH_SIZE, labels_to_balance, epoch_len,
                                          [person], data_mode='Test',contacts=['M'])


        out_callback = OutputPlotCallback(person_dict, trial_dir,
                                          samples_per_label_per_person=10,used_persons=[person], picture_name=person, data_mode='Test',
                                          phase='Train')

        callbacks = [NanCallback(),
            TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
            SaveKerasModelCallback(trial_dir, f"model_trial_{trial.number}"),
            FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1', data_mode = 'Test', proj='pca',
                                     metric="euclidean", picture_name_prefix=person + 'test_dict', used_persons=[person],
                                     num_of_components=3, samples_per_label_per_person=10, phase='Train'),

            out_callback,
        ]



        model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            epochs=epoch_num,
            # steps_per_epoch=n_samples,  # Explicitly set number of steps per epoch
            validation_data=val_ds,
            # validation_steps=n_samples,  # Same for validation
            verbose=2
        )
        metrics_values = model.evaluate(
            val_ds,
            return_dict=True
        )
        try:
            person_mae = metrics_values['mae']
            person_mse = metrics_values['mse']
        except:
            person_mae = metrics_values['multiply_mae']
            person_mse = metrics_values['multiply_mse']
        personal_metrics_dict[person] = {'mae': person_mae, 'mse': person_mse}
    max_val_mae = max([metrics['mae'] for person, metrics in personal_metrics_dict.items()])
    mean_val_mae = np.mean([metrics['mae'] for person, metrics in personal_metrics_dict.items()])

    # Calculate the standard deviation (sigma) of results_for_same_parameters
    # sigma_results = np.std(results_for_same_parameters)

    # write info
    file_name = "info.txt"
    # Create the full file path
    file_path = str(trials_dir) + file_name

    # Write information to the file
    with open(file_path, "a") as file:
        file.write(f"trial{trial.number}\n")
        file.write(f"window size {snc_window_size_hp}\n")

        # file.write(f"persons val loss {persons_val_loss_dict}")
        # file.write("\n")
        file.write(f'personal second stage accuracy {personal_metrics_dict}')
        file.write("\n")
        file.write("\n")
        file.write("\n")
    return mean_val_mae

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
    BATCH_SIZE = 1024
    N_TRIALS = 10
    labels_to_balance = [0 ,0.5 , 1, 2]
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
    file_dir = r"C:\Users\sofia.a\PycharmProjects\DATA_2024\Sorted_old"
    person_dict = get_weight_file_from_dir(file_dir)

    logs_root_dir, log_dir, trials_dir = logging_dirs()

    hint_model_path = r"C:\Users\sofia.a\PycharmProjects\Production\WeightEstimation2K\MODELS\ZeroID\model_trial_2 (1).keras"
    # Create optuna study
    storage_name = os.path.join(f"sqlite:///{logs_root_dir.resolve()}", "wld.db")
    study_name = "attention_weight_classifier_snc" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    study = optuna.create_study(directions=['minimize'], study_name=study_name,
                                sampler=optuna.samplers.NSGAIISampler(),
                                storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=N_TRIALS)

    # Info from the study
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # https://medium.com/optuna/announcing-optuna-3-4-0087644c92fa
    figure = optuna.visualization.plot_optimization_history(study)
    save_plotly_graph_object(study, figure)

    # Save the best epoch for each trial.
    # In multi-objective there is no 'best' trial, we have a tradeoff
    print("Study statistics: ")
    print(" Number of finished trials: ", len(study.trials))
    print(" Number of pruned trials: ", len(pruned_trials))
    print(" Number of complete trials: ", len(complete_trials))