
# from training_playground.constants import *
WINDOW_SIZE =1377#1394# 1620# 648
import tensorflow as tf
# from training_playground.models import create_attention_weight_estimation_model, get_optimizer
import keras
from custom.layers import SEMGScatteringTransform
from google_friendly_model.tangent_proj import TangentSpaceLayer
from google_friendly_model.build_model import SequentialCrossSpectralSpd_matricesLayer

# def create_test_model(window_size_snc=306,  J_snc=5, Q_snc=(2, 1),
#                                           undersampling=4.8,
#                                             units=10, conv_activation='tanh',
#                                             attention_layers_for_one_sensor=1,
#                                              use_time_ordering=False,
#                                                 use_sensor_attention=False,
#                                             use_sensor_ordering=False,
#                                              num_heads=4, key_dim=8, scale_activation='linear',
#                                              optimizer='LAMB', learning_rate=0.0016,
#                                           weight_decay=0.0,max_weight=8, compile=True):
#     # Define inputs to the model
#     input_layer_snc1 = tf.keras.Input(shape=window_size_snc, name='Snc1')
#     input_layer_snc2 = tf.keras.Input(shape=window_size_snc, name='Snc2')
#     input_layer_snc3 = tf.keras.Input(shape=window_size_snc, name='Snc3')
#
#     x1 = tf.keras.layers.Dense(10)(input_layer_snc1)
#     x2 = tf.keras.layers.Dense(10)(input_layer_snc2)
#     x3 = tf.keras.layers.Dense(10)(input_layer_snc3)
#
#     final_dense_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='final_dense')
#     out_1 = max_weight * final_dense_layer(x1)
#     out_2 = max_weight * final_dense_layer(x2)
#     out_3 = max_weight * final_dense_layer(x3)
#
#     out = tf.reduce_mean(tf.concat([tf.expand_dims(out_1,axis=2), tf.expand_dims(out_2,axis=2), tf.expand_dims(out_3,axis=2)], axis=2), axis=2)
#     out = tf.keras.layers.Flatten()(out)
#
#     # inputs = [input_layer_snc1, input_layer_snc2, input_layer_snc3]
#     inputs = {'snc_1':input_layer_snc1, 'snc_2':input_layer_snc2, 'snc_3':input_layer_snc3}
#     model = tf.keras.Model(inputs=inputs,
#                            outputs=[out, out_1, out_2, out_3]
#                                 )
#     if compile:
#         opt = get_optimizer(optimizer=optimizer, learningrestore_rate=learning_rate, weight_decay=weight_decay)
#         model.compile(loss=[None,tf.keras.losses.MeanSquaredError(),tf.keras.losses.MeanSquaredError(),tf.keras.losses.MeanSquaredError()],
#                       metrics=['mae'],
#                       optimizer=opt)
#
#     return model
#
# attention_snc_model_parameters_dict = { 'window_size_snc': WINDOW_SIZE,
#                                         'J_snc': 5,
#                                         'Q_snc': (2, 1),
#                                         'undersampling': 4.4,#trial.suggest_float('undersampling', 3.5, 5.5,
#                                                                     # step=0.15),#4.8
#                                         'use_attention': False, # trial.suggest_categorical('use_attention',[True, False]),
#                                         'attention_layers_for_one_sensor':1,#trial.suggest_int('attention_layers_for_one_sensor', 1, 2),
#                                         'use_sensor_attention': True,# trial.suggest_categorical('use_sensor_attention',[True, False]),
#
#                                         'use_sensor_ordrestoreering':True,# trial.suggest_categorical('use_sensor_ordering', [True, False]),
#                                         'units': 13,#trial.suggest_int('units', 10, 20),
#                                         'conv_activation': 'relu',#trial.suggest_categorical('conv_activation', ['tanh', 'sigmoid', 'relu', 'linear']),
#                                         'use_time_ordering': True,# trial.suggest_categorical('use_time_ordering', [True, False]),
#                                         'num_heads': 3,#trial.suggest_int('num_heads', 3, 4),#4
#                                         'key_dim_for_snc': 30,#trial.suggest_int('key_dim', 10, 30),#6
#                                         'key_dim_for_sensor_att': 10,
#
#                                         # 'optimizer': 'Adam',# trial.suggest_categorical('optimizer', ['LAMB', 'Adam']),#'LAMB',
#                                         # 'weight_decay': 0,# 0.01,#trial.suggest_float('weight_decay', 0.0, 0.1, step=0.01),
#                                         # 'learning_rate': 0.0016,
#                                             }


class OnDeviceModel(tf.Module):

  def __init__(self, model_path):
    # self.model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(input_shape=(IMG_SIZE, IMG_SIZE), name='flatten'),
    #     tf.keras.layers.Dense(128, activation='relu', name='dense_1'),
    #     tf.keras.layers.Dense(10, name='dense_2')
    # ])
    # self.model = create_attention_weight_estimation_model(window_size_snc=WINDOW_SIZE, compile=False)
    # self.model = create_test_model(window_size_snc=WINDOW_SIZE, compile=False)

    #LOADING from KERAS
    # self.model =  create_attention_weight_estimation_model(**attention_snc_model_parameters_dict, compile=False)
    # custom_objects = {
    #     'SEMGScatteringTransform': SEMGScatteringTransform,
    # }
    custom_objects = {
        'SequentialCrossSpectralSpd_matricesLayer': SequentialCrossSpectralSpd_matricesLayer,
        'TangentSpaceLayer': TangentSpaceLayer
    }

    self.model = keras.models.load_model(
        model_path,
        custom_objects=custom_objects,
        compile=False,
        safe_mode=False
    )
    # optimizer ='Adam'
    # learning_rate = 0.0016
    # weight_decay = 0
    opt = 'Adam'#get_optimizer(optimizer=optimizer, learning_rate=learning_rate, weight_decay=weight_decay)
    # self.model.compile(#losses=[None, tf.keras.losses.MeanSquaredError(), tf.keras.losses.MeanSquaredError(),
    #                     #tf.keras.losses.MeanSquaredError()],
    #               loss = {'out': None, 'out1': keras.losses.MeanSquaredError(), 'out2': keras.losses.MeanSquaredError(),
    #                       'out3': keras.losses.MeanSquaredError()},
    #               metrics=['mae'],
    #               optimizer=opt)

    self.model.compile(loss=keras.losses.Huber(),
                      # loss_weights=loss_weights,
                      metrics=['mae', 'mse'],
                      optimizer=opt,
                      run_eagerly=True)
    # self.model.compile(
    #     optimizer='sgd',
    #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))


  # The `train` function takes a batch of input images and labels.
  @tf.function(input_signature=[
      tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
      tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
      tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
      # [tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32)],
      tf.TensorSpec([None, 1], tf.float32)
  ])
  def train(self, snc1, snc2, snc3, y):

    with tf.GradientTape() as tape:
      # prediction,prediction1,prediction2,prediction3 = self.model({'snc_1': snc1,'snc_2': snc2,'snc_3': snc3})
      prediction = self.model({'snc_1': snc1, 'snc_2': snc2, 'snc_3': snc3})
      loss = self.model.loss(y, prediction)
      # loss = 1/3*(self.model.loss['out1'](y,prediction1)+
      #             self.model.loss['out2'](y,prediction2)+
      #             self.model.loss['out3'](y,prediction3))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))
    result = {"loss": loss}
    return result

  @tf.function(input_signature=[
      tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
  ])

  # def evaluate(self, snc1, snc2, snc3, y):
  #   # prediction,prediction1,prediction2,prediction3 = self.model({'snc_1': snc1,'snc_2': snc2,'snc_3': snc3})
  #
  #   # loss = 1/3*(self.model.loss['out1'](y,prediction1)+
  #   #           self.model.loss['out2'](y,prediction2)+
  #   #           self.model.loss['out3'](y,prediction3))
  #   prediction = self.model({'snc_1': snc1, 'snc_2': snc2, 'snc_3': snc3})
  #   loss = self.model.loss(y, prediction)
  #
  #   result = {"loss": loss}
  #   return result

  # @tf.function(input_signature=[
  #     tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
  # ])

  def infer(self, snc_1,snc_2, snc_3):
    inputs = {'snc_1':snc_1, 'snc_2': snc_2, 'snc_3': snc_3}
    logits = self.model(inputs)
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # return {
    #     "out": logits[0],
    #     "out1": logits[1],
    #     "out2": logits[2],
    #     "out3": logits[3]
    # }
    return {"out": logits}

  # @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  # def save(self, checkpoint_path):
  #   tensor_names = [weight.name for weight in self.model.weights]
  #   tensors_to_save = [weight.read_value() for weight in self.model.weights]
  #   tf.raw_ops.Save(
  #       filename=checkpoint_path, tensor_names=tensor_names,
  #       data=tensors_to_save, name='save')
  #   return {
  #       "checkpoint_path": checkpoint_path
  #   }
  #
  # @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  # def restore(self, checkpoint_path):
  #   restored_tensors = {}
  #   for var in self.model.weights:
  #     restored = tf.raw_ops.Restore(
  #         file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
  #         name='restore')
  #     var.assign(restored)
  #     restored_tensors[var.name] = restored
  #   return restored_tensors

  # CLAUDE SUGGESTION
  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def save(self, checkpoint_path):
      tensor_names = [weight.name for weight in self.model.weights]
      tensors_to_save = [weight.read_value() for weight in self.model.weights]

      # Use the save op with the filename tensor
      tf.raw_ops.Save(
          filename=checkpoint_path,
          tensor_names=tensor_names,
          data=tensors_to_save,
          name='save')

      return {
          "checkpoint_path": checkpoint_path
      }

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def restore(self, checkpoint_path):
      restored_tensors = {}

      for var in self.model.weights:
          # Use restore op for each variable
          restored = tf.raw_ops.Restore(
              file_pattern=checkpoint_path,
              tensor_name=var.name,
              dt=var.dtype,
              name='restore')

          # Assign the restored value to the variable
          var.assign(restored)
          restored_tensors[var.name] = restored

      return restored_tensors


class OnDeviceSubModel(tf.Module):

  def __init__(self, model_path):

    self.model = keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False
    )

    opt = 'Adam'

    self.model.compile(loss=keras.losses.Huber(),
                      metrics=['mae', 'mse'],
                      optimizer=opt,
                      run_eagerly=True)

  # The `train` function takes a batch of input images and labels.
  @tf.function(input_signature=[
      tf.TensorSpec([None, 6], tf.float32),
      # tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
      # tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
      # [tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32), tf.TensorSpec([None, 1], tf.float32)],
      tf.TensorSpec([None, 1], tf.float32)
  ])
  def train(self, tg_vector, y):

    with tf.GradientTape() as tape:
      # prediction,prediction1,prediction2,prediction3 = self.model({'snc_1': snc1,'snc_2': snc2,'snc_3': snc3})
      prediction = self.model({'snc_1': tg_vector})
      loss = self.model.loss(y, prediction)
      # loss = 1/3*(self.model.loss['out1'](y,prediction1)+
      #             self.model.loss['out2'](y,prediction2)+
      #             self.model.loss['out3'](y,prediction3))
    gradients = tape.gradient(loss, self.model.trainable_variables)
    self.model.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))
    result = {"loss": loss}
    return result

  @tf.function(input_signature=[
      tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
  ])

  # def evaluate(self, snc1, snc2, snc3, y):
  #   # prediction,prediction1,prediction2,prediction3 = self.model({'snc_1': snc1,'snc_2': snc2,'snc_3': snc3})
  #
  #   # loss = 1/3*(self.model.loss['out1'](y,prediction1)+
  #   #           self.model.loss['out2'](y,prediction2)+
  #   #           self.model.loss['out3'](y,prediction3))
  #   prediction = self.model({'snc_1': snc1, 'snc_2': snc2, 'snc_3': snc3})
  #   loss = self.model.loss(y, prediction)
  #
  #   result = {"loss": loss}
  #   return result

  # @tf.function(input_signature=[
  #     tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),tf.TensorSpec([None, WINDOW_SIZE], tf.float32),
  # ])

  def infer(self, snc_1,snc_2, snc_3):
    inputs = {'snc_1':snc_1, 'snc_2': snc_2, 'snc_3': snc_3}
    logits = self.model(inputs)
    # probabilities = tf.nn.softmax(logits, axis=-1)
    # return {
    #     "out": logits[0],
    #     "out1": logits[1],
    #     "out2": logits[2],
    #     "out3": logits[3]
    # }
    return {"out": logits}

  # @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  # def save(self, checkpoint_path):
  #   tensor_names = [weight.name for weight in self.model.weights]
  #   tensors_to_save = [weight.read_value() for weight in self.model.weights]
  #   tf.raw_ops.Save(
  #       filename=checkpoint_path, tensor_names=tensor_names,
  #       data=tensors_to_save, name='save')
  #   return {
  #       "checkpoint_path": checkpoint_path
  #   }
  #
  # @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  # def restore(self, checkpoint_path):
  #   restored_tensors = {}
  #   for var in self.model.weights:
  #     restored = tf.raw_ops.Restore(
  #         file_pattern=checkpoint_path, tensor_name=var.name, dt=var.dtype,
  #         name='restore')
  #     var.assign(restored)
  #     restored_tensors[var.name] = restored
  #   return restored_tensors

  # CLAUDE SUGGESTION
  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def save(self, checkpoint_path):
      tensor_names = [weight.name for weight in self.model.weights]
      tensors_to_save = [weight.read_value() for weight in self.model.weights]

      # Use the save op with the filename tensor
      tf.raw_ops.Save(
          filename=checkpoint_path,
          tensor_names=tensor_names,
          data=tensors_to_save,
          name='save')

      return {
          "checkpoint_path": checkpoint_path
      }

  @tf.function(input_signature=[tf.TensorSpec(shape=[], dtype=tf.string)])
  def restore(self, checkpoint_path):
      restored_tensors = {}

      for var in self.model.weights:
          # Use restore op for each variable
          restored = tf.raw_ops.Restore(
              file_pattern=checkpoint_path,
              tensor_name=var.name,
              dt=var.dtype,
              name='restore')

          # Assign the restored value to the variable
          var.assign(restored)
          restored_tensors[var.name] = restored

      return restored_tensors