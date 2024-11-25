import numpy as np

import keras
import keras.ops as K
from custom.losses import ProtoLoss
from models import get_optimizer
def check_for_nans_keras3(tensor):
    return K.any(K.isnan(tensor))
def check_for_nans_numpy(array):
    return np.isnan(array).any()



import keras
import numpy as np

def ex_model(sensor_num=2, window_size_snc=306,

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
    # input_layer_snc1 = tf.keras.Input(shape=(rows, cols), name='Snc1')
    input_layer_snc2 = keras.Input(shape=(window_size_snc,), name='snc_2')
    input_layer_snc3 = keras.Input(shape=(window_size_snc,), name='snc_3')

    x = keras.layers.BatchNormalization()(input_layer_snc2)
    out = keras.layers.Dense(embd_dim, activation=dense_activation,kernel_initializer='he_normal', name = 'person_id_final')(x)

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


def check_for_nans_numpy(array):
    return np.isnan(array).any()

class LayerOutputMonitor(keras.callbacks.Callback):
    def __init__(self, layer_names):
        super().__init__()
        self.layer_names = layer_names
        self.layer_functions = {}

    def set_model(self, model):
        super().set_model(model)
        for layer_name in self.layer_names:
            layer = self.model.get_layer(layer_name)
            self.layer_functions[layer_name] = keras.Function(
                self.model.inputs, [layer.output]
            )

    def on_train_batch_end(self, batch, logs=None):
        x = logs.get('inputs')
        if x is None:
            print("Warning: No inputs available in logs. Cannot check for NaNs.")
            return

        for layer_name, layer_function in self.layer_functions.items():
            layer_output = layer_function(x)
            if tf.executing_eagerly():
                if check_for_nans_tf(layer_output[0]):
                    print(f"NaN found in layer {layer_name} output at batch {batch}")
            else:
                print(f"Warning: Not running eagerly. Cannot check for NaNs in {layer_name}.")

def custom_fit(self, *args, **kwargs):
    callbacks = kwargs.get('callbacks', [])
    layer_output_monitor = LayerOutputMonitor(['scattering_time_domain','person_id'])
    callbacks.append(layer_output_monitor)
    kwargs['callbacks'] = callbacks
    return self.original_fit(*args, **kwargs)

# class LayerOutputMonitor(keras.callbacks.Callback):
#     def __init__(self, layer_names):
#         super().__init__()
#         self.layer_names = layer_names
#
#     def on_batch_end(self, batch, logs=None):
#         for layer_name in self.layer_names:
#             layer = self.model.get_layer(layer_name)
#             layer_output = layer.output
#             # get_layer_output(self.model, persons_input_data, self.layer_name)
#             # layer_output_value = keras.backend.get_value(layer_output)
#             if isinstance(layer_output, list):
#                 layer_output = layer_output[0]
#             if check_for_nans_keras3(layer_output):
#                 print(f"NaN found in layer {layer_name} output at batch {batch}")


class ValueMonitorCallback(keras.callbacks.Callback):
    def __init__(self, monitor_layers=None):
        super().__init__()
        self.monitor_layers = monitor_layers or []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}

        # Monitor output values
        y_pred = self.model.output
        y_pred_value = y_pred.numpy()
        if np.isnan(y_pred_value).any():
            print(f"NaN found in model output at batch {batch}")

        # Monitor gradients
        trainable_weights = self.model.trainable_weights
        with keras.backend.GradientTape() as tape:
            y_pred = self.model(self.model.inputs)
            loss = self.model.compiled_loss(self.model.targets, y_pred)
        gradients = tape.gradient(loss, trainable_weights)
        gradient_values = [g.numpy() for g in gradients if g is not None]

        for i, grad_value in enumerate(gradient_values):
            if np.isnan(grad_value).any():
                print(f"NaN found in gradient of weight {i} at batch {batch}")

        # Monitor specific layer outputs
        for layer_name in self.monitor_layers:
            layer = self.model.get_layer(layer_name)
            layer_output = layer.output
            layer_output_value = layer_output.numpy()
            if np.isnan(layer_output_value).any():
                print(f"NaN found in output of layer {layer_name} at batch {batch}")
def check_for_nans(data_dict):
    for key, value in data_dict['inputs'].items():
        if np.isnan(value).any() or np.isinf(value).any():
            print(f"NaN or Inf found in {key}")
            return True
    if np.isnan(data_dict['labels']).any() or np.isinf(data_dict['labels']).any():
        print("NaN or Inf found in labels")
        return True
    return False


def enhanced_data_check(data_dict):
    has_issues = False

    print("Checking input data:")
    for key, value in data_dict['inputs'].items():
        print(f"\nAnalyzing {key}:")
        print(f"  Shape: {value.shape}")
        print(f"  Data type: {value.dtype}")
        print(f"  Contains NaN: {np.isnan(value).any()}")
        print(f"  Contains Inf: {np.isinf(value).any()}")
        print(f"  Min value: {np.min(value)}")
        print(f"  Max value: {np.max(value)}")
        print(f"  Mean value: {np.mean(value)}")
        print(f"  Standard deviation: {np.std(value)}")

        if np.isnan(value).any() or np.isinf(value).any():
            print(f"  WARNING: NaN or Inf found in {key}")
            has_issues = True

    print("\nChecking labels:")
    labels = data_dict['labels']
    print(f"  Shape: {labels.shape}")
    print(f"  Data type: {labels.dtype}")
    print(f"  Contains NaN: {np.isnan(labels).any()}")
    print(f"  Contains Inf: {np.isinf(labels).any()}")
    print(f"  Unique values: {np.unique(labels)}")

    if np.isnan(labels).any() or np.isinf(labels).any():
        print("  WARNING: NaN or Inf found in labels")
        has_issues = True

    return has_issues


class WeightMonitorCallback(keras.callbacks.Callback):
    """
    Callback to monitor model weights during training.
    Prints statistics about weights at the end of each epoch or batch.
    """

    def __init__(self, print_freq='epoch', layer_wise=True, detailed_stats=True):
        """
        Args:
            print_freq: 'epoch' or 'batch' - when to print stats
            layer_wise: If True, print stats for each layer separately
            detailed_stats: If True, print detailed statistics
        """
        super().__init__()
        self.print_freq = print_freq
        self.layer_wise = layer_wise
        self.detailed_stats = detailed_stats

    def _analyze_weights(self, weights, name=""):
        """Analyze weights and return statistics."""
        stats = {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'min': np.min(weights),
            'max': np.max(weights),
            'zeros': np.sum(weights == 0),
            'total': weights.size
        }

        if self.detailed_stats:
            stats.update({
                'median': np.median(weights),
                'q1': np.percentile(weights, 25),
                'q3': np.percentile(weights, 75),
                'non_zero': np.sum(weights != 0),
                'positive': np.sum(weights > 0),
                'negative': np.sum(weights < 0)
            })

        return stats


    def _print_weight_stats(self, epoch=None, batch=None):
        """Print weight statistics."""
        print("\n" + "=" * 50)
        if epoch is not None:
            print(f"Weight statistics for epoch {epoch}")
        if batch is not None:
            print(f"Weight statistics for batch {batch}")
        print("=" * 50)

        if self.layer_wise:
            # Print stats for each layer
            for layer in self.model.layers:
                weights = layer.get_weights()
                if weights:
                    print(f"\nLayer: {layer.name}")
                    for i, w in enumerate(weights):
                        print(f"\nWeight array {i}:")
                        print(f"Shape: {w.shape}")
                        stats = self._analyze_weights(w)

                        print(f"Mean: {stats['mean']:.6f}")
            print(f"Max:  {stats['max']:.6f}")
            print(f"Zeros: {stats['zeros']}/{stats['total']} "
                  f"({stats['zeros'] / stats['total'] * 100:.2f}%)")

            if self.detailed_stats:
                print(f"Median: {stats['median']:.6f}")
                print(f"Q1:     {stats['q1']:.6f}")
                print(f"Q3:     {stats['q3']:.6f}")
                print(f"Non-zero: {stats['non_zero']}/{stats['total']} "
                      f"({stats['non_zero'] / stats['total'] * 100:.2f}%)")
                print(f"Positive: {stats['positive']}/{stats['total']} "
                      f"({stats['positive'] / stats['total'] * 100:.2f}%)")
                print(f"Negative: {stats['negative']}/{stats['total']} "
                      f"({stats['negative'] / stats['total'] * 100:.2f}%)")

        else:
            # Print stats for all weights combined
            all_weights = np.concatenate([w.flatten() for layer in self.model.layers
                                          for w in layer.get_weights()])
            stats = self._analyze_weights(all_weights)
            print("\nOverall weight statistics:")
            print(f"Total parameters: {stats['total']}")
            print(f"Mean: {stats['mean']:.6f}")
            print(f"Std:  {stats['std']:.6f}")
            print(f"Min:  {stats['min']:.6f}")
            print(f"Max:  {stats['max']:.6f}")
            print(f"Zeros: {stats['zeros']}/{stats['total']} "
                  f"({stats['zeros'] / stats['total'] * 100:.2f}%)")

            if self.detailed_stats:
                print(f"Median: {stats['median']:.6f}")
                print(f"Q1:     {stats['q1']:.6f}")
                print(f"Q3:     {stats['q3']:.6f}")
                print(f"Non-zero: {stats['non_zero']}/{stats['total']} "
                      f"({stats['non_zero'] / stats['total'] * 100:.2f}%)")
                print(f"Positive: {stats['positive']}/{stats['total']} "
                      f"({stats['positive'] / stats['total'] * 100:.2f}%)")
                print(f"Negative: {stats['negative']}/{stats['total']} "
                      f"({stats['negative'] / stats['total'] * 100:.2f}%)")


    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        if self.print_freq == 'epoch':
            self._print_weight_stats(epoch=epoch)


    def on_batch_end(self, batch, logs=None):
        """Called at the end of each batch."""
        if self.print_freq == 'batch':
            self._print_weight_stats(batch=batch)

