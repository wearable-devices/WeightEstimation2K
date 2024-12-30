
from models_dir.user_embeddings import  WeightEstimationWithUserEmbeddings
from db_generators.user_embd_generator import *
from utils.get_data import get_weight_file_from_dir
import os
from datetime import datetime
from pathlib import Path
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from custom.callbacks import *

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
    # SENSOR_NUM = 3
    window_size = 256
    batch_size = 1024
    persons_for_train_initial_model = ['Avihoo', 'Aviner', 'Shai', 'HishamCleaned',
                                       'Alisa','Molham','Michael',
                                                        # 'Liav',
                                                         'Daniel',#'Ofek',
                                                         'Foad', 'Asher2','Itai',#'Perry',
                                       'Tom'
                                       ]
    persons_for_test = [ 'Leeor','Lee'
                        'Liav',
       #                   'Daniel',
       #                   'Foad',
       #                  'Asher2', 'Lee',
       #  #'Guy'
                        ]
    # Create person to index mapping
    persons = persons_for_train_initial_model
    person_to_idx = {name: idx for idx, name in enumerate(persons)}

    # Create the model
    model = WeightEstimationWithUserEmbeddings(max_weight=2, window_size_snc=window_size).model()


    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0016),
        loss='Huber',  # or your custom loss function
        metrics=['mae']  # add relevant metrics
    )

    # USE_PRETRAINED_MODEL=True
    file_dir = '/home/wld-algo-6/Data/Sorted'
    person_dict = get_weight_file_from_dir(file_dir)

    # gr = UserEmbdModelGenerator(person_dict,person_to_idx, window_size, data_mode='all', batch_size=1024,
    #                              person_names=persons_for_train_initial_model, labels_to_balance=[0,0.5,1,2], epoch_len=None,
    #                              contacts=['M'])
    # gr.__getitem__(0)

    labels_to_balance = [0, 0.5, 1, 2]
    epoch_len = 100#15
    train_ds = create_ds_for_embd_model(person_dict, person_to_idx, window_size=window_size, batch_size=batch_size, labels_to_balance=labels_to_balance,
                                        epoch_len = epoch_len,
                                        data_mode='Train',
                                        person_names=persons_for_train_initial_model,  contacts=['M'])
    val_ds = create_ds_for_embd_model(person_dict, person_to_idx, window_size=window_size, batch_size=batch_size, labels_to_balance=labels_to_balance,
                                        epoch_len=epoch_len,
                                        data_mode='Test',
                                        person_names=persons_for_train_initial_model, contacts=['M'])

    # train_generator = UserEmbdModelGenerator(
    #     person_dict,
    #     person_to_idx,
    #     window_size=window_size,
    #     batch_size=batch_size,
    #     data_mode='Train',
    #     labels_to_balance=labels_to_balance,
    #     epoch_len=epoch_len,
    #     person_names=persons_for_train_initial_model,
    #     contacts=['M']
    # )
    # data =train_generator.__getitem__(0)
    # model(data[0])
    model.summary()

    #DEBUG
    # train_generator = UserEmbdModelGenerator(
    #     person_dict,
    #     person_to_idx,
    #     window_size=window_size,
    #     batch_size=4,
    #     data_mode='Train',
    #     labels_to_balance=labels_to_balance,
    #     epoch_len=epoch_len,
    #     person_names=['Leeor'],
    #     contacts=['M']
    # )
    # data = train_generator.__getitem__(0)
    # model(data[0])


    model.fit(
        train_ds,
        batch_size=batch_size,
        callbacks=[ModelCheckpoint(
            filepath=os.path.join(log_dir,
                                  # f"pre_trained_model_trial_{trial.number}__epoch_{{epoch:03d}}.weights.h5"
                                  f'model.weights.h5'),
            # Added .weights.h5
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'),
            TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard')),
            SaveKerasModelCallback(log_dir, f'model', phase='train')
            # MetricsTrackingCallback()
        ],
        epochs=500,#2000,
        validation_data=val_ds,
        verbose=1,
    )

    new_user = 'Leeor'
    train_ds = create_ds_for_embd_model(person_dict, {new_user: 19}, window_size=window_size, batch_size=batch_size,
                                        labels_to_balance=labels_to_balance, epoch_len=epoch_len,
                                        data_mode='Train',
                                        person_names=[new_user], contacts=['M'])
    val_ds = create_ds_for_embd_model(person_dict, {new_user: 19}, window_size=window_size, batch_size=batch_size,
                                      labels_to_balance=labels_to_balance,
                                      epoch_len=epoch_len,
                                      data_mode='Test',
                                      person_names=[new_user], contacts=['M'])

    # train_generator = UserEmbdModelGenerator(
    #     person_dict,
    #     person_to_idx = {new_user: 19},
    #     window_size=window_size,
    #     batch_size=512,
    #     data_mode='Train',
    #     labels_to_balance=labels_to_balance,
    #     epoch_len=epoch_len,
    #     person_names=[new_user],
    #     contacts=['M']
    # )
    # data = train_generator.__getitem__(0)
    # model(data[0])

    model.fit(
        train_ds,
        batch_size=batch_size,
        callbacks=[ModelCheckpoint(
            filepath=os.path.join(log_dir,
                                  # f"pre_trained_model_trial_{trial.number}__epoch_{{epoch:03d}}.weights.h5"
                                  f'model.weights.h5'),
            # Added .weights.h5
            verbose=1,
            save_weights_only=True,
            save_freq='epoch'),
            TensorBoard(log_dir=os.path.join(log_dir, 'tensorboard')),
            SaveKerasModelCallback(log_dir, f'model', phase='train')
            # MetricsTrackingCallback()
        ],
        epochs=50,  # 2000,
        validation_data=val_ds,
        verbose=1,
    )