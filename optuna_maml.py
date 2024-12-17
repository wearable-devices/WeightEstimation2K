import tensorflow as tf
import keras
from keras import layers
import numpy as np
from custom.layers import *
import keras.ops as K
from models_dir.maml import create_maml_model, MAML
from db_generators.maml_generator import *
from utils.get_data import get_weight_file_from_dir

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


# Example usage with MNIST
# def prepare_mnist_data():
#     """Prepare MNIST dataset for few-shot learning"""
#     (x_train, y_train), _ = keras.datasets.mnist.load_data()
#     x_train = x_train.astype('float32') / 255.0
#     x_train = np.expand_dims(x_train, axis=-1)
#     return x_train, y_train
#
#
# # Setup and training
# x_train, y_train = prepare_mnist_data()


if __name__ == "__main__":
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
    task_generator.generate_batch()
    # Create and train MAML
    maml = MAML(
        model_fn=create_maml_model,
        inner_learning_rate=0.01,
        outer_learning_rate=0.001
    )

    # Train the model
    train_maml(
        maml,
        task_generator,
        num_epochs=15,
        tasks_per_epoch=10,
        eval_interval=1
    )

    # Example of adapting to a new task
    support_x, support_y, query_x, query_y = task_generator.sample_task()
    adapted_params = maml.adapt_to_task(support_x, support_y)

    # Save original parameters
    original_params = [tf.identity(w) for w in maml.model.trainable_variables]

    # Update model with adapted parameters
    for w, w_adapted in zip(maml.model.trainable_variables, adapted_params):
        w.assign(w_adapted)

    # Make predictions on query set
    predictions = maml.model(query_x)

    # Restore original parameters
    for w, w_original in zip(maml.model.trainable_variables, original_params):
        w.assign(w_original)
