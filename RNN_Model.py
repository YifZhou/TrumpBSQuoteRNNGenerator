# pylint: disable=unused-import,g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class TrumpBSModel(object):

  def __init__(self, vocabularySize, config_param):
    self.vocabularySize = vocabularySize
    self.config = config_param

    self._inputX = tf.placeholder(tf.int32, [self.config.batch_size, self.config.sequence_size], "InputsX")
    self._inputTargetsY = tf.placeholder(tf.int32, [self.config.batch_size, self.config.sequence_size], "InputTargetsY")


    #Converting Input in an Embedded form
    with tf.device("/cpu:0"): #Tells Tensorflow what GPU to use specifically
      embedding = tf.get_variable("embedding", [self.vocabularySize, self.config.embeddingSize])
      embeddingLookedUp = tf.nn.embedding_lookup(embedding, self._inputX)
      inputs = tf.split(1, self.config.sequence_size, embeddingLookedUp)
      inputTensorsAsList = [tf.squeeze(input_, [1]) for input_ in inputs]


    #Define Tensor RNN
    singleRNNCell = rnn_cell.BasicRNNCell(self.config.hidden_size)
    self.multilayerRNN =  rnn_cell.MultiRNNCell([singleRNNCell] * self.config.num_layers)
    self._initial_state = self.multilayerRNN.zero_state(self.config.batch_size, tf.float32)

    #Defining Logits
    hidden_layer_output, states = rnn.rnn(self.multilayerRNN, inputTensorsAsList, initial_state=self._initial_state)
    hidden_layer_output = tf.reshape(tf.concat(1, hidden_layer_output), [-1, self.config.hidden_size])
    self._logits = tf.nn.xw_plus_b(hidden_layer_output, tf.get_variable("softmax_w", [self.config.hidden_size, self.vocabularySize]), tf.get_variable("softmax_b", [self.vocabularySize]))
    self._predictionSoftmax = tf.nn.softmax(self._logits)

    #Define the loss
    loss = seq2seq.sequence_loss_by_example([self._logits], [tf.reshape(self._inputTargetsY, [-1])], [tf.ones([self.config.batch_size * self.config.sequence_size])], self.vocabularySize)
    self._cost = tf.div(tf.reduce_sum(loss), self.config.batch_size)

    self._final_state = states[-1]


  def defineTensorGradientDescent(self):
    self._learningRate = tf.Variable(0.0, trainable=False)

    trainingVars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainingVars),self.config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self.learningRate)
    self._tensorGradientDescentTrainingOperation = optimizer.apply_gradients(zip(grads, trainingVars))


  def assign_learningRate(self, session, lr_value):
    session.run(tf.assign(self.learningRate, lr_value))

  @property
  def inputX(self):
    return self._inputX

  @property
  def inputTargetsY(self):
    return self._inputTargetsY

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def final_state(self):
    return self._final_state

  @property
  def learningRate(self):
    return self._learningRate

  @property
  def gradient_desc_training_op(self):
    return self._tensorGradientDescentTrainingOperation

  @property
  def predictionSoftmax(self):
    return self._predictionSoftmax

  @property
  def logits(self):
    return self._logits



