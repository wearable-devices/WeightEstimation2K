import os
import optuna
from datetime import datetime
from optuna_dashboard import save_plotly_graph_object
from optuna.trial import TrialState
from pathlib import Path
import keras
from psd_models import mpf_model
from optuna_snc import fit_and_take_the_best, count_parameters
from db_generators.generators import  create_data_for_model
from keras.callbacks import TensorBoard
from custom.psd_layers import SequentialCrossSpectralDensityLayer_pyriemann
from custom.callbacks import *

from utils.get_data import get_weight_file_from_dir
from models import get_optimizer, get_loss


def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    # Define the search space and sample parameter values
    snc_window_size_hp = 1394#1754 #trial.suggest_int("snc_window_size", 800, 1800, step=18)  #162 1044#128#648#
    addition_weight_hp = 0#trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)
    epoch_num =  20#15
    epoch_len = 10#5  # None
    use_pretrained_model = False# trial.suggest_categorical('use_pretrained_model',[True, False])

    batch_size_np = 1024  # trial.suggest_int('batch_size', 512, 2048, step=512)
    # train_ds = train_ds.take(5)

    # pruning_callback = optuna.integration.tensorboard.TensorBoardCallback(trial)
    trial_dir = trials_dir / f"trial_{trial.number}"  # Specific trial
    trial_dir.mkdir(exist_ok=True)
    trial_dir = str(trial_dir)
    trial.set_user_attr("directory", trial_dir)  # Attribute can be seen in optuna-dashboard

    # # Data for feature space embedding visualization
    # data_vis_embed = val_ds.as_numpy_iterator().next()
    model_name = f"model_trial_{trial.number}"
    # All callbacks for this trial

    model = mpf_model(window_size=snc_window_size_hp, base_point_calculation='identity',
                      preprocessing=None,#'scattering',
                      frame_length=snc_window_size_hp, frame_step=8,
                      middle_dense_units = 5,# trial.suggest_int("middle_dense_units", 3, 5, step=1),
                      dense_activation = 'relu', #trial.suggest_categorical('dense_activation', ['linear','relu', 'sigmoid']),
                      max_weight=2 + addition_weight_hp,
                      optimizer='Adam', learning_rate=0.016,
                      loss='Huber'
                      )
    model_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/26-05-2025-11-13-31/mobile_friendly_model.keras'
    custom_objects = {'SequentialCrossSpectralDensityLayer_pyriemann': SequentialCrossSpectralDensityLayer_pyriemann}
    model = keras.models.load_model(model_path, custom_objects=custom_objects,
                                    compile=False,
                                    safe_mode=False)
    opt = get_optimizer(optimizer='Adam', learning_rate=0.016)
    loss = get_loss('Huber')
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=['mae'],
    )

    model.summary()
    total_params, trainable_params, non_trainable_params = count_parameters(model)

    train_ds = create_data_for_model(persons_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                     used_persons=persons_for_train_initial_model, data_mode='Train', contacts=['M'])
    val_ds = train_ds

    # train the model
    if use_pretrained_model:
        pretrained_model = fit_and_take_the_best(model, train_ds, val_ds, trial_dir, number=trial.number,
                          model_name='model_best', epochs=25,#30, 15
                                             save_checkp=False)
    else:
        pretrained_model = model
    # Save Keras pretrained or initialized model
    initial_model_path = os.path.join(trial_dir, 'initial_pre_trained_model' + '.keras')
    pretrained_model.save(initial_model_path, save_format='keras')

    # test model
    personal_metrics_dict = {}
    for person in persons_for_test:
        print(f'Training on {person}')
        # take a model
        custom_objects = {'SequentialCrossSpectralDensityLayer_pyriemann': SequentialCrossSpectralDensityLayer_pyriemann}
        model = keras.models.load_model(initial_model_path, custom_objects=custom_objects,
                                        compile=True,
                                        safe_mode=False)


        # Define train and test sets
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                         [person], data_mode='Train', contacts=['M'])
        val_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                       [person], data_mode='Test', contacts=['M'])

        out_callback_test = OutputPlotCallback(person_dict, trial_dir,
                                               samples_per_label_per_person=10, used_persons=[person],
                                               picture_name=person + 'test', data_mode='Test',
                                               phase='Train')
        out_callback_train = OutputPlotCallback(person_dict, trial_dir,
                                                samples_per_label_per_person=10, used_persons=[person],
                                                picture_name=person + 'train', data_mode='Train',
                                                phase='Train')




        callbacks = [TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
                     # FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1', data_mode='Test',
                     #                          proj='none',
                     #                           metric="euclidean", picture_name_prefix=person + 'test_dict', used_persons=[person],
                     #                           num_of_components=1, samples_per_label_per_person=10, phase='test'),
                     out_callback_test,
                     out_callback_train  # out_2d_callback
                     ]

        person_mae_list = []
        model.fit(
            train_ds,
            # batch_size=BATCH_SIZE,
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
        mae_key = [key for key in metrics_values.keys() if 'mae' in key.lower()][0]
        person_mae = metrics_values[mae_key]
        person_mae_list.append(person_mae)
        person_mae = np.mean(person_mae_list)
        # mse_key = [key for key in metrics_values.keys() if 'mse' in key.lower()][0]
        # person_mse = metrics_values[mse_key]

        personal_metrics_dict[person] = {'mae': person_mae,
                                         # 'mse': person_mse
                                         }

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
        file.write(f"model  {model_name} total params  "
                   f"{total_params}.\n")
        file.write(f"model  {model_name} trainable params  "
                   f"{trainable_params}.\n")
        file.write(f"model  {model_name} non-parameters params  "
                   f"{non_trainable_params}.\n")
        # file.write(f"persons val loss {persons_val_loss_dict}")
        # file.write("\n")
        file.write(f'personal second stage accuracy {personal_metrics_dict}')
        file.write("\n")
        file.write("\n")

    return mean_val_mae  # max_val_mae#max_val_loss#mean_val_mse  # max_val_loss


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
    N_TRIALS = 100
    labels_to_balance = [0, 0.5, 1, 2]
    # labels_to_balance = [0,  0.5]
    max_weight = 2
    persons_for_train_initial_model = ['Avihoo', 'Aviner', 'Shai', 'HishamCleaned',
                                       # 'Alisa',
                                       'Molham', 'Daniel',
                                      'Foad',
                                     'Asher2',
                                       #new
                                       'Michael',
                                       'Perry',
                                       'Ofek',
                                       'Tom'
                                       ]
    persons_for_test = ['Alisa',
                        'Liav','Itai','Leeor',
       #
                                         'Lee',
                         # 'Michael',
                         #'Perry',
                   # 'Ofek',
       # 'Tom', #'Guy'
                        ]
    persons_for_plot = persons_for_test

    # USE_PRETRAINED_MODEL=True
    file_dir = '/media/wld-algo-6/Storage/Data/Sorted'#'/home/wld-algo-6/Data/Sorted'
    person_dict = get_weight_file_from_dir(file_dir)

    persons_dict_M = {user: {weight: [dict for dict in list if dict['contact'] == 'M'] for weight, list in person_dict[user].items()} for user in person_dict.keys()}
    persons_dict = persons_dict_M
    logs_root_dir, log_dir, trials_dir = logging_dirs()

    # file_path = str(trials_dir) + 'general_info.txt'


    # Create optuna study
    storage_name = os.path.join(f"sqlite:///{logs_root_dir.resolve()}", "wld.db")
    study_name = "spd_weight_classifier_snc" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
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