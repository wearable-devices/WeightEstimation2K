import keras
from coremltools.converters.mil.mil.ops.defs.iOS15 import softmax
from keras.src.metrics import CategoricalCrossentropy

from custom.layers import *
import keras.ops as K
from models import get_optimizer, get_loss
from db_generators.generators import preprocess_person_data
from keras.callbacks import TensorBoard
from custom.for_debug import check_for_nans, ValueMonitorCallback, \
                LayerOutputMonitor, custom_fit, ex_model,enhanced_data_check
from pathlib import Path
from datetime import datetime
from custom.losses import ProtoLoss
from custom.callbacks import FeatureSpacePlotCallback, NanCallback,SaveKerasModelCallback
import os
from db_generators.generators import create_data_for_userId_model

from utils.get_data import get_weight_file_from_dir
import optuna
from optuna.trial import TrialState
from optuna_dashboard import save_plotly_graph_object

def create_person_zeroid_model(sensor_num=2, window_size_snc=306,
                                             J_snc=5, Q_snc=(2, 1),
                                             undersampling=2,#4.8
                                             units=10, dense_activation='relu',
                                            scattering_type='old',
                                            embd_dim=5,

                                            number_of_persons=10,
                                             optimizer='Adam', learning_rate=0.0016,
                                             weight_decay=0.01,  compile=True,
                                             ):
    '''sensor_fusion could be 'early, attention or mean'''
    # Define inputs to the model
    input_layer_snc1 = keras.Input(shape=(window_size_snc,), name='snc_1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    if scattering_type == 'old':
        scattering_layer = ScatteringTimeDomain(J=J_snc, Q=Q_snc, undersampling=undersampling, max_order=2)
    elif scattering_type == 'SEMG':
        scattering_layer = SEMGScatteringTransform()

    scattered_snc1, scattered_snc11 = scattering_layer(input_layer_snc1)
    scattered_snc2, scattered_snc22 = scattering_layer(input_layer_snc2)
    scattered_snc3, scattered_snc33 = scattering_layer(input_layer_snc3)


    if scattering_type == 'old':
        scattered_snc1 = K.squeeze(scattered_snc1, axis=-1)
        scattered_snc2 = K.squeeze(scattered_snc2, axis=-1)
        scattered_snc3 = K.squeeze(scattered_snc3, axis=-1)

        scattered_snc1 = K.transpose(scattered_snc1, axes=(0, 2, 1))
        scattered_snc2 = K.transpose(scattered_snc2, axes=(0, 2, 1))
        scattered_snc3 = K.transpose(scattered_snc3, axes=(0, 2, 1))

    all_sensors = [scattered_snc1,scattered_snc2,scattered_snc3]
    x = all_sensors[sensor_num-1]

    x = K.mean(x, axis=1)

    x = keras.layers.Dense(units, activation=dense_activation, name = 'dense_1')(x)
    x = keras.layers.Dense(units//2, activation=dense_activation, name='person_id')(x)
    # out = keras.layers.Dense(embd_dim, activation='softmax', name = 'person_distr')(x)
    out = keras.layers.Dense(embd_dim, activation=dense_activation, name = 'person_id_final')(x)




    inputs = {'snc_1': input_layer_snc1, 'snc_2': input_layer_snc2, 'snc_3': input_layer_snc3}
    model = keras.Model(inputs=inputs,
                           outputs=out
                           )
    if compile:
        opt = get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)

        model.compile(loss=ProtoLoss(number_of_persons=number_of_persons),  #'categorical_crossentropy',
                      #metrics=['accuracy',
                               # keras.metrics.Precision(name='precision'),
                               # keras.metrics.Recall(name='recall'),
                               # keras.metrics.F1Score(name='f1_score')
                              # ],
                      optimizer=opt,
                      run_eagerly=True)


    return model
# Create a custom callback to handle multi-value metrics
class CustomTensorBoard(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Convert non-scalar values to scalars if needed
        logs = {k: v.item() if hasattr(v, 'item') else v for k, v in logs.items()}
        super().on_epoch_end(epoch, logs)
def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    snc_window_size_hp = 1548#trial.suggest_int("snc_window_size", 162, 1800, step=18)
    model_parameters_dict = {'sensor_num': 2,
                             'window_size_snc': snc_window_size_hp,
                            'undersampling':2,#4.8
                            'units':14,#trial.suggest_int('units', 8, 25),
                             'dense_activation': 'relu',
                             'scattering_type':'SEMG',
                            'embd_dim': 9,#trial.suggest_int('embd_dim', 5, 15),

                            'number_of_persons':num_persons,
                             'optimizer':'Adam', 'learning_rate':0.0016,
                             'weight_decay':0.0,  'compile':True}

    # pruning_callback = optuna.integration.tensorboard.TensorBoardCallback(trial)
    trial_dir = trials_dir / f"trial_{trial.number}"  # Specific trial
    trial_dir.mkdir(exist_ok=True)
    trial_dir = str(trial_dir)
    trial.set_user_attr("directory", trial_dir)  # Attribute can be seen in optuna-dashboard
    model_name = f"model_trial_{trial.number}"

    model = create_person_zeroid_model(**model_parameters_dict)
    model.summary()

    train_ds = create_data_for_userId_model(person_zero_dict, person_to_idx, snc_window_size_hp, batch_size,
                                            epoch_len, persons,
                                            data_mode='Train', contacts=['M'])


    val_ds = create_data_for_userId_model(person_zero_dict, person_to_idx, snc_window_size_hp, batch_size,
                                          epoch_len, persons,
                                          data_mode='Test', contacts=['M'])
    callbacks = [TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
                 SaveKerasModelCallback(trial_dir, f"model_trial_{trial.number}", phase='train'),
                 NanCallback(),  # ValueMonitorCallback(monitor_layers=['person_id', 'person_id_final']),
                 FeatureSpacePlotCallback(person_dict, trial_dir, layer_name='person_id_final',
                                          data_mode='Test',
                                          proj='tsne',
                                          metric="euclidean", picture_name_prefix='test_dict',
                                          used_persons=persons, task='user_id',
                                          num_of_components=2, samples_per_label_per_person=15, phase='Train')
                 ]

    # Train the model
    model.fit(
        train_ds,
        validation_data=val_ds,
        callbacks=callbacks,
        epochs=50,
        batch_size=batch_size
    )

    metrics_values = model.evaluate(
        val_ds,
        return_dict=True
    )

    val_loss = metrics_values['loss']
    return val_loss

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
    logs_root_dir, log_dir, trials_dir = logging_dirs()
    # file_dir = r"C:\Users\sofia.a\PycharmProjects\DATA_2024\Sorted"#'/home/wld-algo-6/DataCollection/Data'
    file_dir = '/home/wld-algo-6/Data/Sorted'
    person_dict = get_weight_file_from_dir(file_dir)
    person_zero_dict = {person_name: weight_dict[0] for person_name, weight_dict in person_dict.items() if 0 in weight_dict.keys()}

    # snc_window_size = 512
    # Preprocess data
    persons = ['Alisa', 'Asher2', 'Avihoo', 'Aviner', 'HishamCleaned','Lee',
               'Leeor',
               'Daniel',
               'Liav',
                'Foad',
               'Molham' ,
               'Ofek', 'Shai', 'Itai', 'Michael','Perry','Tom'
               ]#,'Guy']

    # Create person to index mapping
    person_to_idx = {name: idx for idx, name in enumerate(persons)}
    num_persons = len(person_to_idx) if persons == 'all' else len(persons)

    epoch_len = 30  # None
    batch_size = 512
    # train_ds = create_data_for_userId_model(person_zero_dict, person_to_idx, snc_window_size, batch_size,
    #                                         epoch_len, persons,
    #                                         data_mode='Train', contacts=['M'])
    # val_ds = create_data_for_userId_model(person_zero_dict, person_to_idx, snc_window_size, batch_size,
    #                                       epoch_len, persons,
    #                                       data_mode='Test', contacts=['M'])





    # Convert labels to one-hot encoding
    num_persons = len(person_to_idx) if persons == 'all' else len(persons)
    # train_labels_onehot = keras.utils.to_categorical(train_data['labels'], num_persons)
    # test_labels_onehot = keras.utils.to_categorical(test_data['labels'], num_persons)

    # Create and train model
    # model = create_person_zeroid_model(
    #                     sensor_num=2,  # Use sensor 2 data
    #                     # scattering_type='SEMG',
    #                     units=10,
    #                     number_of_persons=num_persons,
    #                     window_size_snc=snc_window_size, embd_dim=4,
    #                     dense_activation='relu',
    #                     learning_rate=0.0016,
    #                     # Match your input size
    # )
    #
    # callbacks = [TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard')),
    #              #LayerOutputMonitor([ 'scattering_time_domain','person_id']),
    #              NanCallback(),#ValueMonitorCallback(monitor_layers=['person_id', 'person_id_final']),
    #              FeatureSpacePlotCallback(person_dict, log_dir, layer_name='person_id_final', data_mode='Test',
    #                                       proj='tsne',
    #                                       metric="euclidean", picture_name_prefix='test_dict',
    #                                       used_persons=persons,task='user_id',
    #                                       num_of_components=2, samples_per_label_per_person=15, phase='Train')
    # ]
    #
    # # Train the model
    # history = model.fit(
    #     train_ds,
    #     validation_data=val_ds,
    #     callbacks=callbacks,
    #     epochs=30,
    #     batch_size=batch_size
    # )

    N_TRIALS = 100
    # Create optuna study
    storage_name = os.path.join(f"sqlite:///{logs_root_dir.resolve()}", "wld.db")
    study_name = "attention_weight_classifier_snc_for_user_zeroID" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
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
