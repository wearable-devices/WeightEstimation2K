from imageio.core.imopen import imopen

from db_generators.maml_generator import sample_person_signal_windows
import numpy as np
import keras
import tensorflow as tf

class UserEmbdModelGenerator(keras.utils.Sequence):
    def __init__(self, persons_dict,person_to_idx, window_size=256, data_mode='all', batch_size=1024,
                 person_names=None, labels_to_balance=None, epoch_len=None,
                 contacts=['L', 'M', 'R']):
        self.persons_dict = persons_dict
        self.person_to_idx = person_to_idx
        self.window_size = window_size
        self.batch_size = batch_size
        self.person_names = list(self.persons_dict.keys()) if person_names is None else person_names
        self.labels_to_balance = labels_to_balance
        self.epoch_len = epoch_len
        self.phase = data_mode
        self.contacts = contacts

        # Calculate samples per label per person
        num_labels = len(self.labels_to_balance)  # if self.labels_to_balance else len(set(label for person in self.processed_data.values() for label in person.keys()))
        self.samples_per_label_per_person = self.batch_size // (len(self.person_names) * num_labels)

    def __len__(self):
        # Assuming all persons have the same number of files
        try:
            weight_rec_num = len(
                self.persons_dict[self.person_names[0]][0.5])  # quantity of persons records for each weight
        except:
            weight_rec_num = 1
        try:
            window_num = (
                        1 + (len(self.persons_dict[self.person_names[0]][1][0]['snc_1']) - self.window_size) // 18)
        except:
            ttt = 1
        if self.epoch_len is None:
            epoch_len = int(window_num * weight_rec_num / self.samples_per_label_per_person)
        else:
            epoch_len = self.epoch_len
        return epoch_len  # sum(len(files) for files in self.processed_data[self.person_names[0]].values())

    def __getitem__(self, idx):

        snc1_batch = []
        snc2_batch = []
        snc3_batch = []
        users_batch = []
        labels = []


        for person_name in self.person_names:
            if self.phase == 'all':
                person_data = self.persons_dict[person_name]
            else:
                try:
                    person_data = {weight: [record for record in records if record['phase'] == self.phase] for
                                   weight, records in self.persons_dict[person_name].items()}
                except:
                    print(person_name)
                    print(self.persons_dict[person_name])
            try:
                person_data = {weight: [record for record in records if record['contact'] in self.contacts] for
                               weight, records in person_data.items()}
            except:
                print('contact', person_name,
                      {weight: [record for record in records if record['contact'] in self.contacts] for
                       weight, records in person_data.items()})

            labels_to_use = self.labels_to_balance if self.labels_to_balance else person_data.keys()

            for label in labels_to_use:
                if label not in person_data.keys():
                    continue  # Skip if no data for this label

                for _ in range(self.samples_per_label_per_person):
                    # Randomly select a file for this label
                    try:
                        file_idx = tf.random.uniform([], 0, len(person_data[label]), dtype=tf.int32)
                    except:
                        print(f'Problrm with {person_name}  {person_data[label]}')
                        continue
                    file_data = person_data[label][file_idx.numpy()]

                    if tf.shape(file_data['snc_1'])[0] - self.window_size + 1 > 0:
                        # Generate a random starting point within the file
                        start = tf.random.uniform([], 0, tf.shape(file_data['snc_1'])[0] - self.window_size + 1,
                                                  dtype=tf.int32)

                        # Extract the window
                        snc1_batch.append(file_data['snc_1'][start:start + self.window_size])
                        snc2_batch.append(file_data['snc_2'][start:start + self.window_size])
                        snc3_batch.append(file_data['snc_3'][start:start + self.window_size])
                        users_batch.append(self.person_to_idx[person_name])
                        labels.append(label)
                    else:
                        tf.print('Short file ', person_name, label)
                # But return as a tuple of (inputs_dict, outputs_list)
        inputs = {
            'snc_1': tf.stack(snc1_batch),
            'snc_2': tf.stack(snc2_batch),
            'snc_3': tf.stack(snc3_batch),
            'users': tf.stack(users_batch)
        }

        outputs = [
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(labels),
            tf.convert_to_tensor(labels)
        ]

        return inputs, outputs

def convert_embd_generator_to_dataset(generator):
    def gen_wrapper():
        for i in range(len(generator)):
            inputs, outputs = generator[i]
            # Convert list outputs to tuple
            yield (
                {
                    'snc_1': inputs['snc_1'],
                    'snc_2': inputs['snc_2'],
                    'snc_3': inputs['snc_3'],
                    'users':inputs['users']
                },
                tuple(outputs)  # Convert list to tuple
            )

    # Define the output signature
    output_signature = (
        # Input signatures
        {
            'snc_1': tf.TensorSpec(shape=(None, generator.window_size), dtype=tf.float32),
            'snc_2': tf.TensorSpec(shape=(None, generator.window_size), dtype=tf.float32),
            'snc_3': tf.TensorSpec(shape=(None, generator.window_size), dtype=tf.float32),
            'users':  tf.TensorSpec(shape=(None,), dtype=tf.int32)
        },
        # Output signatures (4 identical outputs)
        (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )

    return tf.data.Dataset.from_generator(
        gen_wrapper,
        output_signature=output_signature
    )

def create_ds_for_embd_model(person_dict, person_to_idx, window_size, data_mode='all', batch_size=1024,
                             person_names=None, labels_to_balance=None, epoch_len=None,
                             contacts=['L', 'M', 'R']):
    # Create datasets
    train_generator = UserEmbdModelGenerator(
        person_dict,
        person_to_idx,
        window_size=window_size,
        batch_size=batch_size,
        data_mode=data_mode,
        labels_to_balance=labels_to_balance,
        epoch_len=epoch_len,
        person_names=person_names,
        contacts=contacts
      )
    train_generator.__getitem__(0)
    train_ds = convert_embd_generator_to_dataset(train_generator)
    return train_ds