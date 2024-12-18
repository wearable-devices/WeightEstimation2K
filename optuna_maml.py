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
from custom.callbacks import OutputPlotCallback
from pathlib import Path
from datetime import datetime


def train_maml(
        maml: MAML,
        task_generator: TaskGenerator,
        num_epochs: int = 1000,
        tasks_per_epoch: int = 100,
        eval_interval: int = 10
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

    # Training loop
    for epoch in range(num_epochs):
        epoch_losses = []

        # Train on tasks_per_epoch batches
        for _ in range(tasks_per_epoch):
            batch = next(train_generator)
            loss = maml.outer_loop(batch)
            epoch_losses.append(loss)

        # Calculate mean loss for the epoch
        mean_loss = tf.reduce_mean(epoch_losses)

        # Print progress
        if (epoch + 1) % eval_interval == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {mean_loss:.4f}")

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

    # Create task generator
    task_generator = TaskGenerator(
        person_dict,
        n_way=4,  # 4-way classification
        k_shot=1,  # 1-shot learning
        q_queries=15,  # 15 query examples per class
        batch_size=6  # 6 tasks per batch
    )
    # task_generator.sample_task()
    # task_generator.generate_batch()
    # Create and train MAML
    maml = MAML(
        model_fn=create_maml_model,
        inner_learning_rate=0.001,
        outer_learning_rate=0.001,
        inner_steps=10
    )

    # Train the model
    train_maml(
        maml,
        task_generator,
        num_epochs=20,
        tasks_per_epoch=10,
        eval_interval=1
    )

    # Example of adapting to a new task
    user = 'Lee'

    # Adaptation loop
    adapted_params = None
    best_loss = float('inf')
    losses = []
    for i in range(100):
        support_snc1, support_snc2, support_snc3, support_labels, query_snc1, query_snc2, query_snc3, query_labels = task_generator.sample_task(
            user)
        support_data = [support_snc1, support_snc2, support_snc3]

        # Adapt
        updated_parameters, loss = maml.adapt_to_task(support_data, support_labels)
        losses.append(loss)

        # Keep best parameters
        if loss < best_loss:
            best_loss = loss
            best_params = [tf.identity(w) for w in updated_parameters]

        # Update model with current best parameters
        if best_params is not None:
            for w, w_adapted in zip(maml.model.trainable_variables, best_params):
                w.assign(w_adapted)

        if i % 10 == 0:
            print(f'Iteration {i} {user}, Current loss: {loss:.4f}, Best loss: {best_loss:.4f}')

    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title(f'Adaptation Loss for {user}')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.show()

    # Save original parameters
    original_params = [tf.identity(w) for w in maml.model.trainable_variables]

    # Update model with adapted parameters
    for w, w_adapted in zip(maml.model.trainable_variables, adapted_params):
        w.assign(w_adapted)

    # Make predictions on query set
    query_data = [query_snc1, query_snc2, query_snc3]
    predictions = maml.model(query_data)

    # Restore original parameters
    for w, w_original in zip(maml.model.trainable_variables, original_params):
        w.assign(w_original)

    ds = create_data_for_model(person_dict, 256, 1024, [0,0.5,1,2], 5,
                               [user], data_mode='Test', contacts=['M'])

    out_callback = OutputPlotCallback(person_dict, log_dir,
                                      samples_per_label_per_person=10, used_persons=[user], picture_name=user,
                                      data_mode='Test',
                                      phase='Train')

    callbacks = [#NanCallback(),
                 # TensorBoard(log_dir=os.path.join(trial_dir, 'tensorboard')),
                 # SaveKerasModelCallback(trial_dir, f"model_trial_{trial.number}"),
                out_callback,  # out_2d_callback
        ]
    maml.model.evaluate(
        ds,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=2
    )


