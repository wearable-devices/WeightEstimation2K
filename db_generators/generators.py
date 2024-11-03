import tensorflow as tf
import  numpy as np
import random

class MultiInputGenerator(tf.keras.utils.Sequence):
    def __init__(self, persons_dict, window_size, phase='all', batch_size=1024, person_names=None, labels_to_balance=None, epoch_len=None):
        self.persons_dict = persons_dict
        self.window_size = window_size
        self.batch_size = batch_size
        self.person_names = list(self.persons_dict.keys()) if person_names is None else person_names
        self.labels_to_balance = labels_to_balance or []
        self.epoch_len = epoch_len
        self.phase = phase

        # Calculate samples per label per person
        num_labels = len(self.labels_to_balance) #if self.labels_to_balance else len(set(label for person in self.processed_data.values() for label in person.keys()))
        self.samples_per_label_per_person = self.batch_size // (len(self.person_names) * num_labels)

    def __len__(self):
        # Assuming all persons have the same number of files
        weight_rec_num = len(self.persons_dict[self.person_names[0]][1000]) # quantity of persons records for each weight
        try:
            window_num = (1+(len(self.persons_dict[self.person_names[0]][1][0]['snc_1'])-self.window_size)//18)
        except:
            ttt = 1
        if self.epoch_len is None:
            epoch_len = int(window_num*weight_rec_num/self.samples_per_label_per_person)
        else:
            epoch_len = self.epoch_len
        return epoch_len#sum(len(files) for files in self.processed_data[self.person_names[0]].values())

    def __getitem__(self, idx):

        snc1_batch = []
        snc2_batch = []
        snc3_batch = []
        labels = []

        for person_name in self.person_names:
            if self.phase == 'all':
                person_data = self.persons_dict[person_name]
            else:
                person_data = [record for record in self.persons_dict[person_name] if record['phase'] == self.phase]

            labels_to_use = self.labels_to_balance if self.labels_to_balance else person_data.keys()

            for label in labels_to_use:
                if label not in person_data.keys():
                    continue  # Skip if no data for this label

                for _ in range(self.samples_per_label_per_person):
                    # Randomly select a file for this label
                    file_idx = tf.random.uniform([], 0, len(person_data[label]), dtype=tf.int32)
                    file_data = person_data[label][file_idx.numpy()]

                    if tf.shape(file_data['snc_1'])[0] - self.window_size + 1>0:
                        # Generate a random starting point within the file
                        start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - self.window_size + 1,
                                                  dtype=tf.int32)

                        # Extract the window
                        snc1_batch.append(file_data['snc_1'][start:start + self.window_size])
                        snc2_batch.append(file_data['snc_2'][start:start + self.window_size])
                        snc3_batch.append(file_data['snc_3'][start:start + self.window_size])
                        labels.append(label)
                    else:
                        tf.print('Short file ', person_name, label)

        return [tf.stack(snc1_batch),
                tf.stack(snc2_batch),
                tf.stack(snc3_batch)], tf.convert_to_tensor(labels)



class MatchingLearningGenerator(tf.keras.utils.Sequence):
    '''generator for create_personal_weight_estimation_model'''
    def __init__(self, persons_dict_for_labeled_data, persons_dict_for_query_data, window_size, k_shot, q_query,
                 samples_per_weight, persons_per_batch, tasks_per_person, labeled_window_size=None):
        self.persons_dict_for_labeled_data = persons_dict_for_labeled_data
        self.persons_dict_for_query_data = persons_dict_for_query_data
        self.window_size = window_size
        self.k_shot = k_shot
        self.q_query = q_query
        self.samples_per_weight = samples_per_weight # how many instances of each weight will be taken for labeled input
        self.persons_per_batch = persons_per_batch
        self.tasks_per_person = tasks_per_person
        self.labeled_window_size = window_size
        if labeled_window_size is not  None:
            self.labeled_window_size = labeled_window_size
        self.person_names = list(self.persons_dict_for_labeled_data.keys())
        self.labels = list(set(label for person in self.persons_dict_for_labeled_data.values() for label in person.keys()))

    def __len__(self):
        return 100  # or however many batches you want per epoch

    def __getitem__(self, idx):

        # choose persons
        persons_in_the_batch = random.sample(self.person_names, self.persons_per_batch)
        labeled_snc1_tensors, labeled_snc2_tensors, labeled_snc3_tensors = [], [], []
        query_snc1, query_snc2, query_snc3 = [], [], []
        support_labels, query_labels = [], []
        for person in persons_in_the_batch:
            person_weights = list(set(self.labels) & set(self.persons_dict_for_labeled_data[person]))
            if len(person_weights) < 6:
                print(f'person {person} has not all data')
            for _ in range(self.tasks_per_person):
                labeled_snc1_list, labeled_snc2_list, labeled_snc3_list = [], [], []
                person_support_labels=[]
                for label in sorted(person_weights):
                    person_label_data = self.persons_dict_for_labeled_data[person][label]
                    try:
                        person_query_data = self.persons_dict_for_query_data[person][label]
                    except:
                        print(f'no {label} weight for {person}')
                    # n_examples = self.k_shot + self.q_query
                    # n_examples = self.samples_per_weight + 1 #1 for query
                    file_indices = np.random.choice(len(person_label_data), self.samples_per_weight, replace=True)
                    labeled_and_query_snc_1, labeled_and_query_snc_2, labeled_and_query_snc_3 = [], [], []
                    query_snc_1, query_snc_2, query_snc_3 = [], [], []

                    for i, file_idx in enumerate(file_indices):
                        file_data = person_label_data[file_idx]
                        start = np.random.randint(0, len(file_data['snc_1']) - self.window_size + 1)
                        snc1 = file_data['snc_1'][start:start + self.labeled_window_size]
                        snc2 = file_data['snc_2'][start:start + self.labeled_window_size]
                        snc3 = file_data['snc_3'][start:start + self.labeled_window_size]

                        # if i < self.sampels_per_weight:
                        labeled_and_query_snc_1.append(tf.expand_dims(snc1, axis=0)) # now it is only for labeled
                        labeled_and_query_snc_2.append(tf.expand_dims(snc2, axis=0))
                        labeled_and_query_snc_3.append(tf.expand_dims(snc3, axis=0))
                        # person_support_labels.append(tf.expand_dims(label, axis=0))
                    labeled_snc_1_with_same_label = tf.concat(labeled_and_query_snc_1, axis=0)
                    labeled_snc_2_with_same_label = tf.concat(labeled_and_query_snc_2, axis=0)
                    labeled_snc_3_with_same_label = tf.concat(labeled_and_query_snc_3, axis=0)
                    labeled_snc1_list.append(labeled_snc_1_with_same_label)
                    labeled_snc2_list.append(labeled_snc_2_with_same_label)
                    labeled_snc3_list.append(labeled_snc_3_with_same_label)
                    person_support_labels.extend([label]*self.samples_per_weight)


                    # choose query
                    file_indices = np.random.choice(len(person_query_data), 1, replace=True)
                    for i, file_idx in enumerate(file_indices):
                        file_data = person_label_data[file_idx]
                        start = np.random.randint(0, len(file_data['snc_1']) - self.window_size + 1)
                        snc1 = file_data['snc_1'][start:start + self.window_size]
                        snc2 = file_data['snc_2'][start:start + self.window_size]
                        snc3 = file_data['snc_3'][start:start + self.window_size]

                        query_snc_1.append(tf.expand_dims(snc1, axis=0))  # now it is only for query
                        query_snc_2.append(tf.expand_dims(snc2, axis=0))
                        query_snc_3.append(tf.expand_dims(snc3, axis=0))
                    query_snc_1_with_same_label = tf.concat([labeled_and_query_snc_1[-1]], axis=0)
                    query_snc_2_with_same_label = tf.concat([labeled_and_query_snc_2[-1]], axis=0)
                    query_snc_3_with_same_label = tf.concat([labeled_and_query_snc_3[-1]], axis=0)
                    query_snc1.append(query_snc_1_with_same_label)
                    query_snc2.append(query_snc_2_with_same_label)
                    query_snc3.append(query_snc_3_with_same_label)
                    query_labels.extend([label]*1)
                labeled_snc1 = tf.concat(labeled_snc1_list, axis=0)
                labeled_snc2 = tf.concat(labeled_snc2_list, axis=0)
                labeled_snc3 = tf.concat(labeled_snc3_list, axis=0)

                labeled_snc1_tensors.extend([labeled_snc1]*len(person_weights))
                labeled_snc2_tensors.extend([labeled_snc2]*len(person_weights))
                labeled_snc3_tensors.extend([labeled_snc3]*len(person_weights))
                person_support_labels = tf.convert_to_tensor(person_support_labels)
                support_labels.extend([person_support_labels]*len(person_weights))

        return ([tf.stack(query_snc1), tf.stack(query_snc2),tf.stack(query_snc3),
                 tf.stack(labeled_snc1_tensors), tf.stack(labeled_snc2_tensors), tf.stack(labeled_snc3_tensors), tf.expand_dims(tf.stack(support_labels), axis=-1)],
                [tf.stack(query_labels), tf.stack(query_labels), tf.stack(query_labels), tf.stack(query_labels)])



class OneSncGenerator(tf.keras.utils.Sequence):
    '''generator for create_personal_ embedding_model'''
    def __init__(self, persons_dict, window_size, samples_per_label_per_person=5, sensor_num=1):
        self.persons_dict = persons_dict
        self.window_size = window_size
        self.sensor_num = sensor_num
        self.samples_per_label_per_person = samples_per_label_per_person

        # self.persons_per_batch = persons_per_batch
        # self.tasks_per_person = tasks_per_person
        # self.labels = list(set(label for person in self.persons_dict_for_labeled_data.values() for label in person.keys()))

    def __len__(self):
        return 100  # or however many batches you want per epoch

    def __getitem__(self, idx):

        snc_batch = []
        labels = []

        for person_name in self.persons_dict.keys():
            person_data = self.persons_dict[person_name]
            labels_to_use =  person_data.keys()

            for label in labels_to_use:
                if label not in person_data.keys():
                    continue  # Skip if no data for this label

                for _ in range(self.samples_per_label_per_person):
                    # Randomly select a file for this label
                    file_idx = tf.random.uniform([], 0, len(person_data[label]), dtype=tf.int32)
                    file_data = person_data[label][file_idx.numpy()]

                    # Generate a random starting point within the file
                    start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - self.window_size + 1,
                                              dtype=tf.int32)

                    # Extract the window
                    snc_batch.append(file_data[f'snc_{self.sensor_num}'][start:start + self.window_size])
                    labels.append(label)

        return [tf.stack(snc_batch)], tf.convert_to_tensor(labels)
