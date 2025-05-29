from datetime import datetime
from pathlib import Path
from utils.get_data import get_weight_file_from_dir
from google_friendly_model.build_model import mpf_model, SequentialCrossSpectralDensityLayer_pyriemann
from optuna_snc import count_parameters
from db_generators.generators import  create_data_for_model
from optuns_psd_model import fit_and_take_the_best
import os
import keras
from custom.callbacks import *
from keras.callbacks import TensorBoard

def logging_dirs():
    package_directory = Path(__file__).parent

    logs_root_dir = package_directory / 'logs'
    logs_root_dir.mkdir(exist_ok=True)
    log_dir = package_directory / 'logs' / datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    log_dir.mkdir(exist_ok=True)
    return logs_root_dir, log_dir

if __name__ == "__main__":
    labels_to_balance = [0, 0.5, 1, 2]
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
                                         'Lee',
                         # 'Michael',
                         #'Perry',
                   # 'Ofek',
       # 'Tom', #'Guy'
                        ]
    file_dir = '/media/wld-algo-6/Storage/Data/Sorted'
    person_dict = get_weight_file_from_dir(file_dir)

    persons_dict_M = {user: {weight: [dict for dict in list if dict['contact'] == 'M'] for weight, list in person_dict[user].items()} for user in person_dict.keys()}
    persons_dict = persons_dict_M

    logs_root_dir, log_dir = logging_dirs()

    # Create model
    snc_window_size_hp = 1394  # 1754 #trial.suggest_int("snc_window_size", 800, 1800, step=18)  #162 1044#128#648#
    addition_weight_hp = 0  # trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)

    model = mpf_model(window_size=snc_window_size_hp, base_point_calculation='identity',
                      frame_length=snc_window_size_hp, frame_step=8,
                      middle_dense_units=5,  # trial.suggest_int("middle_dense_units", 3, 5, step=1),
                      dense_activation='relu',
                      # trial.suggest_categorical('dense_activation', ['linear','relu', 'sigmoid']),
                      max_weight=2 + addition_weight_hp,
                      optimizer='Adam', learning_rate=0.016,
                      loss='Huber'
                      )
    # get summary
    model.summary()
    total_params, trainable_params, non_trainable_params = count_parameters(model)

    # Train model
    use_pretrained_model = True
    epoch_num = 20  # 15
    epoch_len = 10  # 5

    batch_size_np = 1024  # trial.suggest_int('batch_size', 512, 2048, step=512)
    # train_ds = train_ds.take(5)

    train_ds = create_data_for_model(persons_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                     used_persons=persons_for_train_initial_model, data_mode='Train', contacts=['M'])
    # val_ds = train_ds

    # train initial model
    if use_pretrained_model:
        pretrained_model = fit_and_take_the_best(model, train_ds, train_ds, log_dir, #number=trial.number,
                                                 model_name='model_best', epochs=25,  # 30, 15
                                                 save_checkp=False)
    else:
        pretrained_model = model

    # Save Keras pretrained or initialized model
    initial_model_path = os.path.join(log_dir, 'initial_pre_trained_model' + '.keras')
    pretrained_model.save(initial_model_path, save_format='keras')

    # test model
    personal_metrics_dict = {}
    for person in persons_for_test:
        print(f'Training on {person}')
        # take a model
        custom_objects = {
            'SequentialCrossSpectralDensityLayer_pyriemann': SequentialCrossSpectralDensityLayer_pyriemann}
        model = keras.models.load_model(initial_model_path, custom_objects=custom_objects,
                                        compile=True,
                                        safe_mode=False)

        # Define train and test sets
        train_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                         [person], data_mode='Train', contacts=['M'])
        val_ds = create_data_for_model(person_dict, snc_window_size_hp, batch_size_np, labels_to_balance, epoch_len,
                                       [person], data_mode='Test', contacts=['M'])

        out_callback_test = OutputPlotCallback(person_dict, log_dir,
                                               samples_per_label_per_person=10, used_persons=[person],
                                               picture_name=person + 'test', data_mode='Test',
                                               phase='Train')
        out_callback_train = OutputPlotCallback(person_dict, log_dir,
                                                samples_per_label_per_person=10, used_persons=[person],
                                                picture_name=person + 'train', data_mode='Train',
                                                phase='Train')

        callbacks = [TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard')),
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

        personal_metrics_dict[person] = {'mae': person_mae,
                                         # 'mse': person_mse
                                         }

    max_val_mae = max([metrics['mae'] for person, metrics in personal_metrics_dict.items()])
    mean_val_mae = np.mean([metrics['mae'] for person, metrics in personal_metrics_dict.items()])
    print(f'mean_val mae {mean_val_mae}')