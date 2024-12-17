import tensorflow as tf
import keras
from keras import layers
import numpy as np
from typing import List, Tuple, Iterator
import random

def sample_person_signal_windows(person_data, window_size, label, sample_num=5, person_name=''):
    """
        Samples random windows of signal data from a person's recordings.

        This function takes a person's signal data containing three synchronized channels (snc_1, snc_2, snc_3)
        and randomly samples fixed-size windows from it. It's useful for creating training datasets
        for machine learning models that work with time series data.

        Args:
            person_data (list): List of dictionaries containing signal data files for a person.
                               Each dictionary should have 'snc_1', 'snc_2', and 'snc_3' keys
                               containing the signal data arrays.
            window_size (int): Size of the window to sample (number of time steps).
            label (any): Label to associate with the sampled windows.
            sample_num (int, optional): Number of windows to sample. Defaults to 5.
            person_name (str, optional): Name of the person for logging purposes. Defaults to ''.

        Returns:
            tuple: Four lists containing:
                - snc1_list: List of sampled windows from the first signal channel
                - snc2_list: List of sampled windows from the second signal channel
                - snc3_list: List of sampled windows from the third signal channel
                - labels: List of labels, repeated for each sampled window

        Notes:
            - The function uses TensorFlow random operations for sampling
            - Skips files that are shorter than the window size
            - Handles errors gracefully by logging problematic cases
        """
    snc1_list = []
    snc2_list = []
    snc3_list = []
    labels = []

    for _ in range(sample_num):
        # Randomly select a file
        try:
            file_idx = tf.random.uniform([], 0, len(person_data), dtype=tf.int32)
        except:
            print(f'Problrm with {person_name}  {person_data}')
            continue
        file_data = person_data[file_idx.numpy()]

        if tf.shape(file_data['snc_1'])[0] - window_size + 1 > 0:
            # Generate a random starting point within the file
            start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - window_size + 1,
                                      dtype=tf.int32)
            # Extract the window
            snc1_list.append(file_data['snc_1'][start:start + window_size])
            snc2_list.append(file_data['snc_2'][start:start + window_size])
            snc3_list.append(file_data['snc_3'][start:start + window_size])
            labels.append(label)
        else:
            tf.print('Short file ', person_name)
    return snc1_list, snc2_list, snc3_list, labels


class TaskGenerator:
    def __init__(
            self,
            # x_data: np.ndarray,
            # y_data: np.ndarray,
            person_dict,
            window_size: int = 256,
            n_way: int = 4,
            k_shot: int = 1,
            q_queries: int = 1,
            batch_size: int = 6,
            phase: str = 'Train',
            contact: str = 'M'

    ):
        """
        Initialize the task generator for N-way, K-shot learning

        Args:
            person_dict: dictionary like {person_name: {weight: list of
                                                        dict = {'name': ,
                                                        'snc_1', 'snc_2', 'snc_3',
                                                        'file_path': path,
                                                        'contact': 'M','L' or 'R',
                                                        'phase': 'Train' or 'Test')} }}
            n_way: Number of classes per task
            k_shot: Number of support examples per class
            q_queries: Number of query examples per class
            batch_size: Number of tasks per batch
            phase: Train,Test or all
        """
        self.person_dict = person_dict
        self.window_size = window_size
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queries = q_queries
        self.batch_size = batch_size
        self.phase = phase
        # Get unique classes
        self.classes = [0, 0.5, 1, 2]#np.unique(y_data)
        # self.num_classes = len(self.classes)

    def sample_task(self, user) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a single task (support and query sets) for the user
        Args:
            user: user_name
        Returns:
            support_x: Support set features
            support_y: Support set labels
            query_x: Query set features
            query_y: Query set labels
        """
        user_data_dict = self.person_dict[user]
        # Randomly select N=self.n_way classes for this task
        task_classes = np.random.choice(
            self.classes, size=self.n_way, replace=False
        )

        support_snc1, support_snc2, support_snc3 = [], [], []
        support_labels = []
        query_snc1, query_snc2, query_snc3 = [], [], []
        query_labels = []

        # For each class in the task
        for class_idx, weight in enumerate(task_classes):
            # Get all examples for this class
            class_examples = user_data_dict[weight]#self.class_indices[class_label]
            used_data = [ex for ex in class_examples if ex['phase'] == self.phase and ex['contact'] == 'M']

            # Sample K+Q windows
            snc1_list, snc2_list, snc3_list, _ = sample_person_signal_windows(used_data, self.window_size,
                                                                                   weight,
                                                                                   sample_num=self.k_shot + self.q_queries,
                                                                                   person_name=user)

            # Split into support and query
            support_snc1_weight = snc1_list[:self.k_shot]
            support_snc2_weight = snc2_list[:self.k_shot]
            support_snc3_weight = snc3_list[:self.k_shot]
            query_snc1_weight = snc1_list[self.k_shot:]
            query_snc2_weight = snc2_list[self.k_shot:]
            query_snc3_weight = snc3_list[self.k_shot:]

            # Add to support set
            support_snc1.extend(support_snc1_weight)
            support_snc2.extend(support_snc2_weight)
            support_snc3.extend(support_snc3_weight)
            support_labels.extend([weight] * self.k_shot)

            # Add to query set
            query_snc1.extend(query_snc1_weight)
            query_snc2.extend(query_snc2_weight)
            query_snc3.extend(query_snc3_weight)
            query_labels.extend([weight] * self.q_queries)

        # Convert to arrays and shuffle
        support_snc1 = np.array(support_snc1)
        support_snc2 = np.array(support_snc2)
        support_snc3 = np.array(support_snc3)
        support_labels = np.array(support_labels)
        query_snc1 = np.array(query_snc1)
        query_snc2 = np.array(query_snc2)
        query_snc3 = np.array(query_snc3)
        query_labels = np.array(query_labels)

        # Shuffle support set
        support_indices = np.random.permutation(len(support_labels))
        support_snc1 = support_snc1[support_indices]
        support_snc2 = support_snc2[support_indices]
        support_snc3 = support_snc3[support_indices]
        support_labels = support_labels[support_indices]

        # Shuffle query set
        query_indices = np.random.permutation(len(query_labels))
        query_snc1 = query_snc1[query_indices]
        query_snc2 = query_snc2[query_indices]
        query_snc3 = query_snc3[query_indices]
        query_labels = query_labels[query_indices]

        return support_snc1, support_snc2, support_snc3, support_labels, query_snc1, query_snc2, query_snc3, query_labels

    def generate_batch(self) -> Iterator[List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Generate batches of tasks

        Yields:
            List of (support_x, support_y, query_x, query_y) tuples
        """
        while True:
            # choose tasks(=users) for the batch
            selected_users = random.sample(list(self.person_dict.keys()), self.batch_size)
            batch = [self.sample_task(user) for user in selected_users]
            yield batch


