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
from custom.heatmap_callback import HeatmapMeanPlotCallback
# from db_generators.create_person_dict import *
import keras
from keras.callbacks import ModelCheckpoint
from models_dir.model_fusion import one_sensor_model_fusion


def fit_and_take_the_best(inint_model, train_ds, val_ds, save_dir, number=0, model_name='', epochs=20, save_checkp=True):
    print(f'fit_and_take_the_best for {model_name}')
    save_best_model_sensor_1 = SaveKerasModelCallback(save_dir, model_name+'best', phase='best')
    # best_model_1_path = os.path.join(trial_dir,'model_sensor_1_best' + str(epoch) + '.keras')
    save_checkpoint = ModelCheckpoint(
                       filepath=os.path.join(save_dir,
                                             # f"pre_trained_model_trial_{number}__epoch_{{epoch:03d}}.weights.h5"),
                                             f"{model_name}_trial_{number}__epoch_{{epoch:03d}}.weights.h5"),
                       # Added .weights.h5
                       verbose=1,
                       save_weights_only=True,
                       save_freq='epoch'),
    callbacks = [save_best_model_sensor_1,
                   save_checkpoint,
                   TensorBoard(log_dir=os.path.join(save_dir, 'tensorboard')),
                   MetricsTrackingCallback()
                   ] if save_checkp else [save_best_model_sensor_1, TensorBoard(log_dir=os.path.join(save_dir, 'tensorboard')),
                   MetricsTrackingCallback()
                   ]
    inint_model.fit(
        train_ds,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        epochs=epochs,
        validation_data=val_ds,
        verbose=1,
    )
    model = save_best_model_sensor_1.best_model
    return model

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
    return total_params, trainable_params, non_trainable_params


def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    # Define the search space and sample parameter values
    snc_window_size_hp = 648#trial.suggest_int("snc_window_size", 162, 1800, step=18)  # 1044#
    addition_weight_hp = 0#trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)
    epoch_num =  20#40
    epoch_len = 10#5  # None
    use_pretrained_model = True# trial.suggest_categorical('use_pretrained_model',[True, False])

    # weight_loss_dict_0 = {weight: 1 for i, weight in enumerate([0, 1, 2, 4, 6, 8])}
    # weight_loss_dict_1 = {weight: (i + 1) / 6 for i, weight in enumerate([0, 1, 2, 4, 6, 8])}
    # weight_loss_dict_2 = {weight: ((i + 2) // 2) / 3 for i, weight in enumerate([0, 1, 2, 4, 6, 8])}

    # weight_loss_dicts = [weight_loss_dict_0, weight_loss_dict_1, weight_loss_dict_2]
    # loss_dict_num_hp = 0  # trial.suggest_int('loss_dict_num', 0, 2)

    batch_size_np = 1024  # trial.suggest_int('batch_size', 512, 2048, step=512)
    # train_ds = train_ds.take(5)


    average_sensors_weight_estimation_model_dict = {'window_size_snc': snc_window_size_hp,
                                                    'scattering_type': 'SEMG',# trial.suggest_categorical('scattering_type', ['old',  'SEMG', ]),
                                                        'J_snc': 7,
                                                        'Q_snc': (2, 1),
                                                    'undersampling': 3,#trial.suggest_float('undersampling', 2, 4, step=0.2),#4.8,
                                                    'scattering_max_order': 1,
                                                    'units':12,# trial.suggest_int('units', 4, 12), #9 21
                                                    'dense_activation': 'elu',#trial.suggest_categorical('dense_activation', ['linear',  'relu','tanh' ]),#'relu',#
                                                    'use_attention': True,#trial.suggest_categorical('use_attention', [True, False ]),
                                                        'key_dim_for_time_attention':10,#trial.suggest_int('key_dim_for_time_attention', 4, 12),#5,
                                                        'attention_layers_for_one_sensor': 2,#trial.suggest_int('key_dim_for_time_attention', 2, 3),
                                                        'use_time_ordering': False,
                                                    'final_activation': 'sigmoid',# trial.suggest_categorical('final_activation',['sigmoid', 'tanh']),
                                                    'add_noise': False,# trial.suggest_categorical('add_noise', [True, False ]),
                                                    'optimizer': 'Adam',#trial.suggest_categorical('optimizer',['Adam', 'AdaBelief']),
                                                    'learning_rate': 0.016,#0.0016,
                                                    'weight_decay': 0.0, 'max_weight': max_weight+addition_weight_hp, 'compile': True,
                                                    'loss': 'Huber',# trial.suggest_categorical('loss', ['Huber', 'WeightedHuberLoss'])
                                                    'loss_delta':1.2,#trial.suggest_float('loss_delta', 0.5, 1.1, step=0.1),
                                                    # 'loss_balance': trial.suggest_float('loss_balance', 0.0, 1, step=0.1),
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


    # persons_val_loss_dict = {person: 0 for person in persons_dirs}
    if create_one_sensor_models:
        # model = create_attention_weight_distr_estimation_model(**attention_distr_snc_model_parameters_dict)
        model_sensor_1 = one_sensors_weight_estimation_proto_model(sensor_num=1, **average_sensors_weight_estimation_model_dict)
        model_sensor_2 = one_sensors_weight_estimation_proto_model(sensor_num=2,
                                                                   **average_sensors_weight_estimation_model_dict)
        model_sensor_3 = one_sensors_weight_estimation_proto_model(sensor_num=3,
                                                                   **average_sensors_weight_estimation_model_dict)
        # model_sensor_all = one_sensors_weight_estimation_proto_model(sensor_num='all',
        #                                                            **average_sensors_weight_estimation_model_dict)

        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                         used_persons=persons_for_train_initial_model, data_mode='Train', contacts=['M'])
        val_ds = train_ds
        # train one sensor models
        model_sensor_1 = fit_and_take_the_best(model_sensor_1, train_ds, val_ds, trial_dir, number=trial.number,
                                               model_name='model_sensor_1_best',epochs=20, save_checkp=False)
        model_sensor_2 = fit_and_take_the_best(model_sensor_2, train_ds, val_ds, trial_dir, number=trial.number,
                                               model_name='model_sensor_2_best', epochs=20, save_checkp=False)
        model_sensor_3 = fit_and_take_the_best(model_sensor_3, train_ds, val_ds, trial_dir, number=trial.number,
                                               model_name='model_sensor_3_best', epochs=20, save_checkp=False)
    else:
        model_1_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/21-03-2025-19-20-07/trials/trial_31/model_sensor_1_bestbest18.keras'
        model_2_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/21-03-2025-19-20-07/trials/trial_31/model_sensor_2_bestbest19.keras'
        model_3_path = '/home/wld-algo-6/Production/WeightEstimation2K/logs/21-03-2025-19-20-07/trials/trial_31/model_sensor_3_bestbest13.keras'
        custom_objects = {'SEMGScatteringTransform': SEMGScatteringTransform}
        model_sensor_1 = keras.models.load_model(model_1_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        model_sensor_2 = keras.models.load_model(model_2_path, custom_objects=custom_objects, compile=False, safe_mode=False)
        model_sensor_3 = keras.models.load_model(model_3_path, custom_objects=custom_objects, compile=False, safe_mode=False)


    fusion_type_tp = 'attention'#trial.suggest_categorical('fusion_type', ['attention',  'majority_vote', ])
    fused_layer_name = trial.suggest_categorical('fused_layer_name', ['mean_layer', 'dense_1', 'dense_2', 'final_dense_1' ])#'dense_1'#
    one_more_dense_hp = trial.suggest_categorical('one_more_dense', [True, False ])
    model = one_sensor_model_fusion(model_sensor_1, model_sensor_2, model_sensor_3,
                             fusion_type=fusion_type_tp, fused_layer_name = fused_layer_name,
                             window_size_snc=snc_window_size_hp,
                             trainable=trial.suggest_categorical('trainable', [True, False ]),#True,
                                    one_more_dense = one_more_dense_hp,
                                    embd_before_fusion = False,#trial.suggest_categorical('embd_before_fusion', [True, False ]),
                             optimizer=average_sensors_weight_estimation_model_dict['optimizer'],
                             learning_rate=average_sensors_weight_estimation_model_dict['learning_rate'],
                             compile=True
                             )
    model.summary()
    total_params, trainable_params, non_trainable_params = count_parameters(model)

    if use_pretrained_model:
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                         used_persons=persons_for_train_initial_model, data_mode='Train', contacts=['M'])
        val_ds = train_ds
        # Before training, verify model configuration
        print("Model configuration:")
        print(model.get_config())

        model = fit_and_take_the_best(model, train_ds, val_ds, trial_dir, number=trial.number,
                                           model_name='fused_model_best', epochs=20, save_checkp=True)

    # Save Keras model
    initial_model_path = os.path.join(trial_dir, 'initial_pre_trained_model' + '.keras')
    model.save(initial_model_path, save_format='keras')

    #Convert the model to TFLite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    tflite_model_path = os.path.join(trial_dir, 'model.tflite')
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f'Model saved to {trial_dir}')
    # TEST MODEL
    personal_metrics_dict = {}
    for person in persons_for_test:
        print(f'Training on {person}')
        # attention_snc_model_parameters_dict['max_weight'] = 2.1#max(train_dict[person].keys()) + 0.5

        # model_snc_path = initial_model_path
        # custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain}
        custom_objects = {'SEMGScatteringTransform': SEMGScatteringTransform}
        model = keras.models.load_model(initial_model_path, custom_objects=custom_objects,
                                           compile=True,
                                           safe_mode=False)

        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                          [person], data_mode='Train', contacts=['M'])
        val_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                          [person], data_mode='Test',contacts=['M'])


        out_callback_test = OutputPlotCallback(person_dict, trial_dir,
                                          samples_per_label_per_person=10,used_persons=[person], picture_name=person+'test', data_mode='Test',
                                          phase='Train')
        out_callback_train = OutputPlotCallback(person_dict, trial_dir,
                                               samples_per_label_per_person=10, used_persons=[person],
                                               picture_name=person+'train', data_mode='Train',
                                               phase='Train')

        out_2d_callback = Output_2d_PlotCallback(person_dict, trial_dir,
                                          samples_per_label_per_person=10,used_persons=[person], picture_name=person+'2d',data_mode='Test',
                                          phase='Train')
        callbacks = [NanCallback(),
            TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
            SaveKerasModelCallback(trial_dir, f"{person}_model_trial_{trial.number}"),
                     ModelCheckpoint(
                         filepath=os.path.join(trial_dir,
                                               f"_{person}_{model_name}_trial_{trial.number}__epoch_{{epoch:03d}}.weights.h5"),
                         # Added .weights.h5
                         verbose=1,
                         save_weights_only=True,
                         save_freq=epoch_num),#'epoch'),
            # FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='dense_1', data_mode = 'Test', proj='pca',
            #                          metric="euclidean", picture_name_prefix=person + 'test_dict', used_persons=[person],
            #                          num_of_components=3, samples_per_label_per_person=10, phase='test', task='weight_estimation'),
            #
            # FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='final_dense_1', data_mode='Test', proj='none',
            #                          metric="euclidean", picture_name_prefix=person + 'test_dict', used_persons=[person],
            #                          num_of_components=1, samples_per_label_per_person=10, phase='test'),

            # HeatmapMeanPlotCallback(person_dict, trial_dir, layer_name='mean_layer',ind_0=10, ind_1 = 11, grid_x_min=-0.001, grid_x_max=0.002, grid_step = 0.00005,
            #                          phase='test', add_samples=True, person=person),
            out_callback_test,
                     out_callback_train# out_2d_callback

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
        mae_key = [key for key in metrics_values.keys() if 'mae' in key.lower()][0]
        person_mae = metrics_values[mae_key]
        # mse_key = [key for key in metrics_values.keys() if 'mse' in key.lower()][0]
        # person_mse = metrics_values[mse_key]

        personal_metrics_dict[person] = {'mae':person_mae,
                                         # 'mse': person_mse
                                         }

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
        file.write("\n")

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
    create_one_sensor_models = False
    SENSOR_NUM = 3
    labels_to_balance = [0, 0.5, 1, 2]
    # labels_to_balance = [0, 1, 2]
    max_weight = 2
    persons_for_train_initial_model = ['Avihoo', 'Aviner', 'Shai', 'HishamCleaned',
                                       'Alisa','Molham', 'Daniel',
                                      'Foad',
                                     'Asher2',
                                       #new
                                       'Michael',
                                       'Perry',
                                       'Ofek',
                                       'Tom'
                                       ]
    persons_for_test = [ 'Leeor',
                        'Liav','Itai',
       #
                                         'Lee',
                         # 'Michael',
                         #'Perry',
                   # 'Ofek',
       # 'Tom', #'Guy'
                        ]
    persons_for_plot = persons_for_test

    # USE_PRETRAINED_MODEL=True
    file_dir = '/home/wld-algo-6/Data/Sorted'
    person_dict = get_weight_file_from_dir(file_dir)

    logs_root_dir, log_dir, trials_dir = logging_dirs()

    file_path = str(trials_dir) + 'general_info.txt'

    # Write information to the file
    with open(file_path, "a") as file:
        file.write(f"sensor num {SENSOR_NUM}\n")
        file.write("\n")
        # file.write(f'max val loss {max([metrics['mse'] for person,metrics in personal_metrics_dict.items()])}')
        # file.write("\n")
        file.write("\n")
        file.write("\n")

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


