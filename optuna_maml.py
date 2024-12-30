import tensorflow as tf
import keras
from keras import layers
import numpy as np
from custom.layers import *
import keras.ops as K
from models_dir.maml import create_maml_model, MAML
from db_generators.maml_generator import *
from utils.get_data import get_weight_file_from_dir
from constants import *
from custom.callbacks import OutputPlotCallback, SaveKerasModelCallback
from pathlib import Path
from datetime import datetime
from models import get_loss,get_optimizer
import os
from keras.callbacks import TensorBoard
import optuna
from utils.plotting_functions import plot_curves
from optuna.trial import TrialState
from optuna_dashboard import save_plotly_graph_object


def train_maml(
        maml: MAML,
        task_generator: TaskGenerator,
        num_epochs: int = 1000,
        tasks_per_epoch: int = 100,
        eval_interval: int = 10,
        plot_loss_and_metrics: bool = False,
        save_path=None
):
    """
    Train the MAML model

    Args:
        maml: MAML instance
        task_generator: TaskGenerator instance
        num_epochs: Number of epochs to train
        tasks_per_epoch: Number of task batches per epoch
        eval_interval: Interval for evaluation
    """
    # Create generators
    train_generator = task_generator.generate_batch()

    losses = []
    metrics_dict = {metric_name: [] for metric_name in maml.metrics.keys()}
    # epoch_metrics = []

    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []
        epoch_metrics = {metric_name: [] for metric_name in maml.metrics.keys()}

        # Train on tasks_per_epoch batches
        for _ in range(tasks_per_epoch):
            batch = next(train_generator)
            loss, metrics = maml.outer_loop(batch)
            epoch_losses.append(loss)
            for metric_name in epoch_metrics.keys():
                epoch_metrics[metric_name].append(metrics[metric_name])
            # for i in range(len(metrics)):
            #     epoch_metrics[i].append(metrics[i])

        # Calculate mean loss for the epoch
        mean_loss = tf.reduce_mean(epoch_losses)
        mean_metrics = {metric_name:tf.reduce_mean(epoch_metrics[metric_name]).numpy() for metric_name in epoch_metrics.keys()}

        losses.append(mean_loss)
        for metric_name in epoch_metrics.keys():
            metrics_dict[metric_name].append(mean_metrics[metric_name])
        # epoch_metrics.append(mean_metrics[0])
        # Print progress
        if (epoch + 1) % eval_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss:.4f}, metrics {metrics}")


    if plot_loss_and_metrics:
        plot_curves({'loss':losses, 'mae': metrics_dict['mae']}, show=True, save_path=save_path, picture_name='loss_curve.png')


def fit_model_on_one_user(model, users, window_size, batch_size, labels_to_balance, epoch_len, epoch_num,save_dir, contacts=['M'], model_name='model'):
    train_ds = create_data_for_model(person_dict, window_size, batch_size, labels_to_balance, epoch_len,
                                     users, data_mode='Train', contacts=contacts)
    val_ds = create_data_for_model(person_dict, window_size, batch_size, labels_to_balance, epoch_len,
                                   users, data_mode='Test', contacts=['M'])

    out_callback = OutputPlotCallback(person_dict, save_dir,
                                      samples_per_label_per_person=10, used_persons=users, picture_name=str(users),
                                      data_mode='Test',
                                      phase='Train')
    callbacks = [#NanCallback(),
                 TensorBoard(log_dir=os.path.join(save_dir, 'tensorboard')),
                 SaveKerasModelCallback(log_dir, f"model_user_{str(users)}"),
                 out_callback,  ]
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
    return metrics_values


def adaptation_for_one_user_from_claude(maml,user):
    best_params = None
    best_loss = float('inf')
    train_losses = []
    val_losses = []

    for i in range(20):
        # Get both support and query data
        support_snc1, support_snc2, support_snc3, support_labels, \
            query_snc1, query_snc2, query_snc3, query_labels = task_generator.sample_task(user)

        support_data = [support_snc1, support_snc2, support_snc3]
        query_data = [query_snc1, query_snc2, query_snc3]

        # Adapt on support set
        updated_parameters, train_loss = maml.adapt_to_task(support_data, support_labels)
        train_losses.append(train_loss)

        # Evaluate on query set
        for w, w_updated in zip(maml.model.trainable_variables, updated_parameters):
            w.assign(w_updated)

        query_pred = maml.model(query_data)[0]
        val_loss = tf.reduce_mean(maml.loss(query_labels, query_pred))
        val_losses.append(val_loss)

        # Keep best parameters based on validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            best_params = [tf.identity(w) for w in updated_parameters]

        if i % 10 == 0:
            print(
                f'Iteration {i}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, Best val loss: {best_loss:.4f}')

    # Plot both training and validation losses
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Adaptation Progress for {user}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Save original parameters
    original_params = [tf.identity(w) for w in maml.model.trainable_variables]

    # Update model with adapted parameters
    for w, w_adapted in zip(maml.model.trainable_variables, updated_parameters):
        w.assign(w_adapted)

    # Make predictions on query set
    query_data = [query_snc1, query_snc2, query_snc3]
    predictions = maml.model(query_data)

    # Restore original parameters
    for w, w_original in zip(maml.model.trainable_variables, original_params):
        w.assign(w_original)

def objective(trial):
    # Clear clutter from previous session graphs
    keras.backend.clear_session()

    # Define the search space and sample parameter values
    snc_window_size_hp = 256#trial.suggest_int("snc_window_size", 162, 1800, step=18)  # 1044#
    addition_weight_hp = 0#trial.suggest_float('addition_weight', 0.0, 0.3, step=0.1)
    epoch_num =  40
    epoch_len = 5  # None
    use_pretrained_model = trial.suggest_categorical('use_pretrained_model',[True, False])
    batch_size_hp = 1024
    model_parameters_dict = {'window_size_snc': snc_window_size_hp, 'scattering_type': 'SEMG',
                             'J_snc': 5, 'Q_snc': (2, 1),
                             'undersampling': 3.4,# trial.suggest_float('undersampling', 2.6, 5, step=0.2),#4.8
                             'use_attention': True,
                             'attention_layers_for_one_sensor': 2,
                             'key_dim_for_time_attention': 5,
                             'units': 8,#trial.suggest_int('units', 5, 15),#10
                             'dense_activation': 'relu',
                             'max_weight': 2, 'final_activation': 'sigmoid',
                             'optimizer': 'Adam',
                             'learning_rate': 0.0016, 'weight_decay': 0.0,
                             'loss': 'Huber'}
    # Create and train MAML
    maml = MAML(
        model_fn=create_maml_model,
        inner_learning_rate=0.01,
        outer_learning_rate=0.001,
        inner_steps=10,
        model_parameters_dict=model_parameters_dict
    )
    maml.model.summary()

    trial_dir = trials_dir / f"trial_{trial.number}"  # Specific trial
    trial_dir.mkdir(exist_ok=True)
    trial_dir = str(trial_dir)
    trial.set_user_attr("directory", trial_dir)  # Attribute can be seen in optuna-dashboard

    if use_pretrained_model:
        # Train the model
        train_maml(
            maml,
            task_generator,
            num_epochs=60,
            tasks_per_epoch=10,
            eval_interval=1,
            plot_loss_and_metrics=True,
            save_path=trial_dir,
        )

    # compile model
    opt = get_optimizer(optimizer='Adam', learning_rate=0.001)
    loss = get_loss('Huber')
    maml.model.compile(
        optimizer=opt,
        loss=[loss, loss, loss, loss
              ],
        metrics=['mae', 'mae', 'mae', 'mae'],
    )

    # Save the model
    initial_model_path =os.path.join(trial_dir, 'trained' + '.keras')
    maml.model.save(initial_model_path, save_format='keras')
    print(f'Model saved to {trial_dir}')

    # batch_size = 1024
    labels_to_balance = [0, 0.5, 1, 2]
    # epoch_len = 10
    # epoch_num = 25
    personal_metrics_dict = {}
    for user in persons_for_test:
        model_name = f"model_trial_{trial.number}_user_{user}"
        print(f'Training on {user}')
        # custom_objects = {'ScatteringTimeDomain': ScatteringTimeDomain}
        model = keras.models.load_model(initial_model_path, #custom_objects=custom_objects,
                                        compile=True,
                                        safe_mode=False)

        metrics_values = fit_model_on_one_user(model, [user], snc_window_size_hp, batch_size_hp, labels_to_balance, epoch_len, epoch_num,
                          trial_dir,
                          contacts=['M'], model_name=model_name)

        person_mae = metrics_values['majority_vote_1_mae']
        personal_metrics_dict[user] = {'mae': person_mae, 'mse': 10000}

    mean_val_mae = np.mean([metrics['mae'] for person, metrics in personal_metrics_dict.items()])
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
    logs_root_dir, log_dir, trials_dir = logging_dirs()
    file_dir = '/home/wld-algo-6/Data/Sorted'
    person_dict = get_weight_file_from_dir(file_dir)
    persons_for_test = ['Leeor',
                                         'Liav',
                        'Lee'
                        #                   'Daniel',
                        #                   'Foad',
                        #                  'Asher2',
                        #             'Ofek',
                        # 'Tom', #'Guy'
                        ]

    window_size = 256

    # Create task generator
    task_generator = TaskGenerator(
        person_dict,
        window_size = window_size,
        n_way=4,  # 4-way classification = number of different weights
        k_shot=1,  # 1-shot learning
        q_queries=15,  # 15 query examples per class
        batch_size=8  # 6 tasks per batch = 6 person
    )
    # task_generator.sample_task()
    # task_generator.generate_batch()

    # Create optuna study
    storage_name = os.path.join(f"sqlite:///{logs_root_dir.resolve()}", "wld.db")
    study_name = "attention_maml_weight_classifier_snc" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    study = optuna.create_study(directions=['minimize'], study_name=study_name,
                                sampler=optuna.samplers.NSGAIISampler(),
                                storage=storage_name, load_if_exists=True)
    study.optimize(objective, n_trials=50)

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
    # model_parameters_dict = {'window_size_snc': 256, 'scattering_type':'SEMG',
    #                   'J_snc':5, 'Q_snc':(2, 1),
    #                   'undersampling':4.8,
    #                   'use_attention':True,
    #                   'attention_layers_for_one_sensor':2, 'key_dim_for_time_attention':5,
    #                   'units':10, 'dense_activation':'relu',
    #                   'max_weight':2, 'final_activation':'sigmoid',
    #                   'optimizer':'Adam',
    #                   'learning_rate':0.0016,'weight_decay':0.0,
    #                   'loss': 'Huber'}
    # # Create and train MAML
    # maml = MAML(
    #     model_fn=create_maml_model,
    #     inner_learning_rate=0.01,
    #     outer_learning_rate=0.001,
    #     inner_steps=10,
    #     model_parameters_dict = model_parameters_dict
    # )
    #
    # # Train the model
    # train_maml(
    #     maml,
    #     task_generator,
    #     num_epochs=500,
    #     tasks_per_epoch=10,
    #     eval_interval=1,
    #     plot_loss_and_metrics=True
    # )
    #
    # # Example of adapting to a new task
    # user = 'Lee'
    # # adaptation_for_one_user_from_claude(user)
    #
    # opt = get_optimizer(optimizer='Adam', learning_rate=0.001)
    # loss = get_loss('Huber')
    # maml.model.compile(
    #     optimizer=opt,
    #     loss=[loss, loss, loss, loss
    #           ],
    #     metrics=['mae', 'mae', 'mae', 'mae'],
    #
    # )
    #
    # batch_size = 1024
    # labels_to_balance = [0,0.5,1,2]
    # epoch_len = 10
    # epoch_num = 25
    # save_dir = log_dir
    # fit_model_on_one_user(maml.model, [user], window_size, batch_size, labels_to_balance, epoch_len, epoch_num, save_dir,
    #                       contacts=['M'])
    # fit_model_on_one_user(maml.model, ['Leeor'], window_size, batch_size, labels_to_balance, epoch_len, epoch_num,
    #                       save_dir,
    #                       contacts=['M'])
    # fit_model_on_one_user(maml.model, ['Liav'], window_size, batch_size, labels_to_balance, epoch_len, epoch_num,
    #                       save_dir,
    #                       contacts=['M'])


    # ds = create_data_for_model(person_dict, 256, 1024, [0,0.5,1,2], 5,
    #                            [user], data_mode='Test', contacts=['M'])
    #
    # out_callback = OutputPlotCallback(person_dict, log_dir,
    #                                   samples_per_label_per_person=10, used_persons=[user], picture_name=user,
    #                                   data_mode='Test',
    #                                   phase='Test')
    #
    # callbacks = [out_callback,  # out_2d_callback
    #              ]
    # maml.model.evaluate(
    #     ds,
    #     batch_size=BATCH_SIZE,
    #     callbacks=callbacks,
    #     verbose=2
    # )


