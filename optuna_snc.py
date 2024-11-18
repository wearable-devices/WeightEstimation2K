from cmath import phase

from keras.callbacks import TensorBoard
import optuna
import os

from optuna_dashboard import save_plotly_graph_object, save_note
from datetime import datetime
from optuna.trial import TrialState
from models import *
from pathlib import Path
from db_generators.generators import MultiInputGenerator, convert_generator_to_dataset, create_data_for_model
# from stam import data_mode, used_persons
from utils.get_data import get_weight_file_from_dir
from constants import *
from custom.callbacks import *
# from db_generators.create_person_dict import *
import keras
from keras.callbacks import ModelCheckpoint


def cleanup_after_trial(callbacks):
    for callback in callbacks:
        if hasattr(callback, 'cleanup'):
            callback.cleanup()

    # Force garbage collection
    import gc
    gc.collect()

    # Clear any remaining tensors in memory
    keras.backend.clear_session()


def count_parameters(model):
    trainable_params = np.sum([np.prod(v.shape) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.shape) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    # print(f'Total parameters: {total_params}')
    # print(f'Trainable parameters: {trainable_params}')
    # print(f'Non-trainable parameters: {non_trainable_params}')
    return total_params, trainable_params, non_trainable_params


def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    # Define the search space and sample parameter values
    snc_window_size_hp = trial.suggest_int("snc_window_size", 162, 1800, step=18)  # 1044#
    addition_weight_hp = trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)
    epoch_num =  30
    epoch_len = 5  # None
    use_pretrained_model = True  # trial.suggest_categorical('use_pretrained_model',[True, False])

    # weight_loss_dict_0 = {weight: 1 for i, weight in enumerate([0, 1, 2, 4, 6, 8])}
    # weight_loss_dict_1 = {weight: (i + 1) / 6 for i, weight in enumerate([0, 1, 2, 4, 6, 8])}
    # weight_loss_dict_2 = {weight: ((i + 2) // 2) / 3 for i, weight in enumerate([0, 1, 2, 4, 6, 8])}

    # weight_loss_dicts = [weight_loss_dict_0, weight_loss_dict_1, weight_loss_dict_2]
    # loss_dict_num_hp = 0  # trial.suggest_int('loss_dict_num', 0, 2)

    batch_size_np = 1024  # trial.suggest_int('batch_size', 512, 2048, step=512)
    # train_ds = train_ds.take(5)
    attention_snc_model_parameters_dict = {'window_size_snc': snc_window_size_hp,
                                           'apply_tfp': False,
                                           'J_snc': 7,#trial.suggest_int('J_snc', 5, 7),  # 5,
                                           'Q_snc': (2, 1),
                                           'undersampling': 4.4,
                                           'scattering_max_order':1,
                                           'use_attention': False,
                                           'attention_layers_for_one_sensor': 1,
                                           # 'use_sensor_attention': trial.suggest_categorical('use_sensor_attention', [True, False]),
                                           'use_sensor_ordering': True,
                                           # trial.suggest_categorical('use_sensor_ordering', [True, False]),
                                           'units': 6,#trial.suggest_int('units', 5, 10),  # 13,#15,#80,#
                                           'dense_activation': 'linear',#trial.suggest_categorical('conv_activation', ['linear',  'relu', ]),# trial.suggest_categorical('conv_activation', ['tanh', 'sigmoid', 'relu', 'linear']),#'relu',
                                           'use_time_ordering': True,
                                           # trial.suggest_categorical('use_time_ordering', [True, False]),
                                           'num_heads': 3,  # trial.suggest_int('num_heads', 3, 4),#4
                                           'key_dim_for_snc': 3,  # trial.suggest_int('key_dim', 5, 20),#6
                                           'key_dim_for_sensor_att': 16,
                                           # trial.suggest_int('key_dim_for_sensor_att', 10, 20),#10,80
                                           'num_sensor_attention_heads': 1,
                                           # trial.suggest_int('num_sensor_attention_heads', 1, 5),#2
                                           'final_activation': 'tanh',
                                           # trial.suggest_categorical('final_activation', ['tanh', 'sigmoid']),

                                           'use_probabilistic_app': False,  # True,
                                           'prob_param': {'smpl_rate': 9,  # 49,
                                                          'sigma_for_labels_prob': 0.4},
                                           'apply_noise': False,
                                           'max_weight': 2.1,
                                           # 'stddev': 0.1,# trial.suggest_float('stddev', 0.0, 0.5, step=0.05),#0.1,
                                           'optimizer': 'Adam',# trial.suggest_categorical('optimizer', ['LAMB', 'Adam']),#'LAMB',
                                           'weight_decay': 0,# 0.01,#trial.suggest_float('weight_decay', 0.0, 0.1, step=0.01),
                                           'learning_rate': 0.0016,
                                           # 'normalization_factor': trial.suggest_float('normalization_factor', 1, 4, step=0.5),
                                           # 'weight_loss_multipliers_dict': weight_loss_dicts[loss_dict_num_hp],
                                           'use_weighted_loss': False,
                                           'sensor_fusion': 'attention',# trial.suggest_categorical('sensor_fusion', ['early', 'attention', 'mean']),
                                           }

    attention_distr_snc_model_parameters_dict = {'window_size_snc': snc_window_size_hp,
                                                 'scattering_type': 'old',#trial.suggest_categorical('scattering_type', ['old',  'SEMG', ]),
                                                 'J_snc': 7,  # trial.suggest_int('J_snc', 5, 7),  # 5,
                                                 'Q_snc': (2, 1),
                                                 'undersampling': 4.4,
                                                 'scattering_max_order': 1,
                                                 'use_attention': False,
                                                 'attention_layers_for_one_sensor': 1,
                                                 'use_sensor_ordering': True,
                                                 'units': 6,
                                                 'dense_activation': 'linear',
                                                 'smpl_rate': 9,
                                                 # trial.suggest_categorical('conv_activation', ['linear',  'relu', ]),# trial.suggest_categorical('conv_activation', ['tanh', 'sigmoid', 'relu', 'linear']),#'relu',
                                                 'use_time_ordering': True,
                                                 'num_heads': 3,
                                                 'key_dim_for_snc': 3,
                                                 'key_dim_for_sensor_att': 16,
                                                 'num_sensor_attention_heads': 1,

                                                 'max_sigma':1,#trial.suggest_float('max_sigma', 0.1, 1, step=0.1),
                                                 'final_activation': 'sigmoid',
                                                 'apply_noise': False,
                                                 'max_weight': 2.1,
                                                 'optimizer': 'Adam',
                                                 'weight_decay': 0,
                                                 'learning_rate': 0.0016,
                                                 'loss_balance': 1,#trial.suggest_float('loss_balance', 0.0, 1, step=0.1),
                                                 'loss_normalize': False,#trial.suggest_categorical('loss_normalize', [True, False]),
                                                 'sensor_fusion': 'attention',
                                                 }

    average_sensors_weight_estimation_model_dict = {'window_size_snc': snc_window_size_hp,
                                                    'J_snc': 7, 'Q_snc': (2, 1),
                                                    'undersampling': trial.suggest_float('undersampling', 1, 5, step=0.2),#4.8,
                                                    'scattering_max_order': 1,
                                                    'units': trial.suggest_int('units', 5, 15),
                                                    'dense_activation': trial.suggest_categorical('dense_activation', ['linear',  'relu', ]),
                                                    'use_attention': False,
                                                    'attention_layers_for_one_sensor': 1,
                                                    'use_time_ordering': False,
                                                    'scattering_type': trial.suggest_categorical('scattering_type', ['old',  'SEMG', ]),
                                                    'final_activation': 'tanh',
                                                    'optimizer': 'Adam', 'learning_rate': 0.0016,
                                                    'weight_decay': 0.0, 'max_weight': 2+addition_weight_hp, 'compile': True,
                                                    'loss': 'Huber',# trial.suggest_categorical('loss', ['Huber', 'mse'])
                                                     }

    # pruning_callback = optuna.integration.tensorboard.TensorBoardCallback(trial)
    trial_dir = trials_dir / f"trial_{trial.number}"  # Specific trial
    trial_dir.mkdir(exist_ok=True)
    trial_dir = str(trial_dir)
    trial.set_user_attr("directory", trial_dir)  # Attribute can be seen in optuna-dashboard

    # # Data for feature space embedding visualization
    # data_vis_embed = val_ds.as_numpy_iterator().next()
    model_name = f"model_trial_{trial.number}"
    # All callbacks for this trial
    labels_to_balance = [0, 0.5, 1, 2]

    # persons_val_loss_dict = {person: 0 for person in persons_dirs}
    # model = create_attention_weight_distr_estimation_model(**attention_distr_snc_model_parameters_dict)
    model = create_one_sensors_weight_estimation_model(sensor_num=1, **average_sensors_weight_estimation_model_dict)

    # model = create_rms_weight_estimation_model(**attention_snc_model_parameters_dict)
    model.summary()
    total_params, trainable_params, non_trainable_params = count_parameters(model)


    # # Create datasets


    if use_pretrained_model:
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
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
                MetricsTrackingCallback()
            ],
            epochs=1,#20,
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
        attention_snc_model_parameters_dict['max_weight'] = 2.5#max(train_dict[person].keys()) + 0.5
        if use_pretrained_model:
            model_snc_path = initial_model_path
            custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain}
            model = keras.models.load_model(model_snc_path, custom_objects=custom_objects,
                                               compile=True,
                                               safe_mode=False)
        else:
            model = create_attention_weight_estimation_model(**attention_snc_model_parameters_dict)

        # total_params, trainable_params, non_trainable_params = count_parameters(model)
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                          [person], data_mode='Train', contacts=['M'])
        val_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                          [person], data_mode='Test',contacts=['M'])


        out_callback = OutputPlotCallback(person_dict, trial_dir,
                                          samples_per_label_per_person=10,used_persons=[person], picture_name=person, data_mode='Test',
                                          phase='Train')

        out_2d_callback = Output_2d_PlotCallback(person_dict, trial_dir,
                                          samples_per_label_per_person=10,used_persons=[person], picture_name=person+'2d',data_mode='Test',
                                          phase='Train')
        callbacks = [
            TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
            SaveKerasModelCallback(trial_dir, f"model_trial_{trial.number}"),
            FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1', data_mode = 'Test', proj='pca',
                                     metric="euclidean", picture_name_prefix=person + 'test_dict', used_persons=[person],
                                     num_of_components=3, samples_per_label_per_person=10, phase='Train'),
            # FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1_for_sensor_1', data_mode='Test', proj='pca',
            #                          metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
            #                          num_of_components=3, samples_per_label_per_person=10, phase='Train'),
            # FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1_for_sensor_2', data_mode='Test',
            #                          proj='pca',
            #                          metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
            #                          num_of_components=3, samples_per_label_per_person=10, phase='Train'),
            # FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1_for_sensor_3', data_mode='Test',
            #                          proj='pca',
            #                          metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
            #                          num_of_components=3, samples_per_label_per_person=10, phase='Train'),

            out_callback,# out_2d_callback

        ]

        # To get first batch
        first_batch = next(iter(train_ds))

        # After dataset creation
        # n_samples = 5  # We know this from the debug output

        # Make sure datasets are repeatable
        # train_ds = train_ds.repeat()
        # val_ds = val_ds.repeat()

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
        person_mae = metrics_values['mae']
        person_mse = metrics_values['mse']
        personal_metrics_dict[person] = {'mae':person_mae, 'mse': person_mse}

    max_val_mae = max([metrics['mae'] for person,metrics in personal_metrics_dict.items()])
    mean_val_mae = np.mean([metrics['mae'] for person,metrics in personal_metrics_dict.items()])

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
        # file.write(f'max val loss {max([metrics['mse'] for person,metrics in personal_metrics_dict.items()])}')
        # file.write("\n")
        file.write("\n")
        file.write("\n")

    # max_val_loss = max(last_val_losses)
    # mean_val_loss = sum(last_val_losses) / len(last_val_losses)
    # mean_val_mse = sum(last_mse) / len(last_mse)

    # Clean up after trial
    # cleanup_after_trial([FeatureSpacePlotCallback(test_dict, trial_dir, layer_name='time_attention_for_sensor_1', proj='pca',
    #                                  metric="euclidean", picture_name=person,
    #                                  num_of_components=2, samples_per_label_per_person=10, phase='train'),
    #         FeatureSpacePlotCallback(test_dict, trial_dir, layer_name='tf.math.reduce_mean', proj='pca',
    #                                  metric="euclidean", picture_name=person,
    #                                  num_of_components=2, samples_per_label_per_person=10, phase='train'),
    #         OutputPlotCallback( test_dict, trial_dir,
    #               samples_per_label_per_person=10, picture_name=person,
    #                             phase='train')])

    return mean_val_mae #max_val_mae#max_val_loss#mean_val_mse  # max_val_loss


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
