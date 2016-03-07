# pylint: disable=unused-import,g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# The hyperparameters used in the model:
# - init_scale - the initial scale of the weights
# - learning_rate - the initial value of the learning rate
# - max_grad_norm - the maximum permissible norm of the gradient
# - num_layers - the number of LSTM layers
# - num_steps - the number of unrolled steps of LSTM
# - hidden_size - the number of LSTM units
# - initialLearningRate_max_epoch - the number of epochs trained with the initial learning rate
# - total_max_epoch - the total number of epochs for training
# - keep_prob - the probability of keeping weights in the dropout layer
# - lr_decay - the decay of the learning rate for each epoch after "max_epoch"
# - batch_size - the batch size


def get_config(flagModel):
  print('Using the %s hyperparameter config' % (flagModel))
  if flagModel == "debugWordToken":
    return DebuggingWordTokenConfig()
  elif flagModel == "debugCharToken":
    return DebuggingCharTokenConfig()
  elif flagModel == "small":
    return SmallConfig()
  elif flagModel == "medium":
    return MediumConfig()
  elif flagModel == "large":
    return LargeConfig()
  else:
    raise ValueError("Invalid model: %s", flagModel)


class DebuggingWordTokenConfig(object):
  init_scale = 0.1
  learning_rate = 0.002
  max_grad_norm = 5
  num_layers = 2

  # When setting num_time_steps and batch_size, make sure that: (data_len // batch_size) -1 // num_time_steps is NOT equal to zero otherwise the algorithm won't run because there will be too little data to input
  # Make sure this is true for data_len values of the training but also validation and test input sizes
  num_time_steps = 4 # num_time_steps need to be smaller than batch_len (batch_len = data_len // batch_size) otherwise there is no epoch to go through
  batch_size = 5
  hidden_size = 128
  #Seems that even in char-rnn-tensorflow code he equates them so for the moment I'm keeping them equal
  embeddingSize = hidden_size
  initialLearningRate_max_epoch = 1
  total_max_epoch = 100
  keep_prob = 1.0
  lr_decay = 0.9


class DebuggingCharTokenConfig(object):
  init_scale = 0.1
  learning_rate = 0.002
  max_grad_norm = 5
  num_layers = 2
  num_time_steps = 50 # num_time_steps need to be smaller than batch_len (batch_len = data_len // batch_size) otherwise there is no epoch to go through
  batch_size = 50
  hidden_size = 128
  #Seems that even in char-rnn-tensorflow code he equates them so for the moment I'm keeping them equal
  embeddingSize = hidden_size
  initialLearningRate_max_epoch = 1
  total_max_epoch = 10000
  keep_prob = 1.0
  lr_decay = 0.97


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_time_steps = 20
  batch_size = 19
  hidden_size = 200
  #Seems that even in char-rnn-tensorflow code he equates them so for the moment I'm keeping them equal
  embeddingSize = hidden_size
  initialLearningRate_max_epoch = 4
  total_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5



class MediumConfig(object):
  """Medium config."""
  init_scale = 0.05
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_time_steps = 35
  batch_size = 20
  hidden_size = 650
  #Seems that even in char-rnn-tensorflow code he equates them so for the moment I'm keeping them equal
  embeddingSize = hidden_size
  initialLearningRate_max_epoch = 6
  total_max_epoch = 39
  keep_prob = 0.5
  lr_decay = 0.8


class LargeConfig(object):
  """Large config."""
  init_scale = 0.04
  learning_rate = 1.0
  max_grad_norm = 10
  num_layers = 2
  num_time_steps = 35
  batch_size = 20
  hidden_size = 1500
  #Seems that even in char-rnn-tensorflow code he equates them so for the moment I'm keeping them equal
  embeddingSize = hidden_size
  initialLearningRate_max_epoch = 14
  total_max_epoch = 55
  keep_prob = 0.35
  lr_decay = 1 / 1.15


