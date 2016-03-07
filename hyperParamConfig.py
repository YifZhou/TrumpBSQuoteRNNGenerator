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


def get_config():
  return DebuggingCharTokenConfig()


class DebuggingCharTokenConfig(object):
  init_scale = 0.1
  learning_rate = 0.002
  max_grad_norm = 5
  num_layers = 2
  sequence_size = 50 # num_time_steps need to be smaller than batch_len (batch_len = data_len // batch_size) otherwise there is no epoch to go through
  batch_size = 50
  hidden_size = 128
  #Seems that even in char-rnn-tensorflow code he equates them so for the moment I'm keeping them equal
  embeddingSize = hidden_size
  initialLearningRate_max_epoch = 1
  total_max_epoch = 10000
  keep_prob = 1.0
  lr_decay = 0.97

