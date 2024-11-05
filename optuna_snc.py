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
from tensorflow.keras.callbacks import ModelCheckpoint


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
    epoch_num =  40
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
                                           'J_snc': trial.suggest_int('J_snc', 5, 7),  # 5,
                                           'Q_snc': (2, 1),
                                           'undersampling': 4.4,
                                           'use_attention': False,
                                           'attention_layers_for_one_sensor': 1,
                                           'use_sensor_attention': True,
                                           'use_sensor_ordering': True,
                                           # trial.suggest_categorical('use_sensor_ordering', [True, False]),
                                           'units': trial.suggest_int('units', 10, 20),  # 13,#15,#80,#
                                           'conv_activation': 'relu',
                                           # trial.suggest_categorical('conv_activation', ['linear',  'relu', ]),# trial.suggest_categorical('conv_activation', ['tanh', 'sigmoid', 'relu', 'linear']),#'relu',
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
                                           'max_weight': 2.5,
                                           # 'stddev': 0.1,# trial.suggest_float('stddev', 0.0, 0.5, step=0.05),#0.1,
                                           'optimizer': 'Adam',# trial.suggest_categorical('optimizer', ['LAMB', 'Adam']),#'LAMB',
                                           'weight_decay': 0,# 0.01,#trial.suggest_float('weight_decay', 0.0, 0.1, step=0.01),
                                           'learning_rate': 0.0016,
                                           # 'normalization_factor': trial.suggest_float('normalization_factor', 1, 4, step=0.5),
                                           # 'weight_loss_multipliers_dict': weight_loss_dicts[loss_dict_num_hp],
                                           'use_weighted_loss': False
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
    last_val_losses = []
    last_mse = []
    # persons_val_loss_dict = {person: 0 for person in persons_dirs}
    model = create_attention_weight_estimation_model(**attention_snc_model_parameters_dict)
    # model = create_rms_weight_estimation_model(**attention_snc_model_parameters_dict)
    model.summary()
    total_params, trainable_params, non_trainable_params = count_parameters(model)


    # # Create datasets


    if use_pretrained_model:
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                         used_persons=persons_for_train_initial_model, data_mode='Train')
        val_ds = train_ds
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
    personal_accuracy = {}
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

        # a = {weight: [record for record in records if record['phase'] == 'Train'] for weight, records in person_dict[person].items()}
        # train_ds.__getitem__(0)
        # val_ds.__getitem__(0)

        # x = train_ds.__len__()
        # train_ds.__getitem__(0)

        out_callback = OutputPlotCallback(person_dict, trial_dir,
                                          samples_per_label_per_person=10,used_persons=[person], picture_name=person,data_mode='Test',
                                          phase='Train')
        callbacks = [
            # Create a callback that saves the model's weights every 5 epochs

            TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
            # SaveKerasModelCallback(trial_dir, f"model_trial_{trial.number}"),

            FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense', data_mode = 'Test', proj='pca',
                                     metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
                                     # considered_weights=[0,4,6,8],
                                     num_of_components=3, samples_per_label_per_person=10, phase='Train'),
            FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1_for_sensor_1', data_mode='Test', proj='pca',
                                     metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
                                     num_of_components=3, samples_per_label_per_person=10, phase='Train'),
            FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1_for_sensor_2', data_mode='Test',
                                     proj='pca',
                                     metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
                                     num_of_components=3, samples_per_label_per_person=10, phase='Train'),
            FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1_for_sensor_3', data_mode='Test',
                                     proj='pca',
                                     metric="euclidean", picture_name=person + 'test_dict', used_persons=[person],
                                     num_of_components=3, samples_per_label_per_person=10, phase='Train'),


            out_callback

        ]

        # To get first batch
        first_batch = next(iter(train_ds))

        history = model.fit(
            train_ds,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            epochs=epoch_num,
            validation_data=val_ds,
            verbose=1,
        )

        # Add all items from out_callback.personal_accuracy to personal_accuracy dictionary
        # out_callback.get_second_stage_accuracy()
        # personal_accuracy.update(out_callback.personal_accuracy)

        val_loss = history.history['val_loss']
        # val_mse = max(
        #     [value for key, value in history.history.items() if key.startswith('val_') and key.endswith('mse')])
        last_val_loss = val_loss[-1]
        # last_val_mse = val_mse[-1]
        min_val_loss = min(val_loss)
        last_val_losses.append(last_val_loss)
        # last_mse.append(last_val_mse)
        # persons_val_loss_dict[person] = last_val_loss

        # max_val_loss = max(last_val_losses)
        # mean_val_loss = sum(last_val_losses) / len(last_val_losses)
        # mean_val_mse = sum(last_mse) / len(last_mse)
        # results_for_same_parameters.append(mean_val_mse)

    # Calculate the standard deviation (sigma) of results_for_same_parameters
    sigma_results = np.std(results_for_same_parameters)

    # write info
    file_name = "info.txt"
    # Create the full file path
    file_path = str(trials_dir) + file_name

    # Open the file in write mode
    with open(file_path, "a") as file:
        # Write information to the file
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
        file.write(f'personal second stage accuracy {personal_accuracy}')
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

    return last_val_loss#mean_val_mse  # max_val_loss


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
    persons_for_train_initial_model = ['Avihoo', 'Aviner', 'Shai']
    persons_for_test = [ 'Leeor',

                       'Liav', 'Daniel', 'Foad',
                        'Asher2', 'Lee',
                   'Ofek',
        #'Tom', 'Guy'
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
