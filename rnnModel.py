# pylint: disable=unused-import,g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
from tensorflow.models.rnn import rnn
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class CharRNNModel(object):

  def __init__(self, model_name, vocabularySize, is_training, config_param):
    self.model_name = model_name
    self.vocabularySize = vocabularySize
    self.config = config_param
    self._inputX = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_time_steps], "InputsX")
    self._inputTargetsY = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_time_steps], "InputTargetsY")


    inputTensorsAsList = self.prepareInputInBatchedFormWithEmbeddings(self.config)


    #Define Tensor RNN
    singleRNNCell = rnn_cell.BasicRNNCell(self.config.hidden_size)
    self.multilayerRNN =  rnn_cell.MultiRNNCell([singleRNNCell] * self.config.num_layers)
    self._initial_state = self.multilayerRNN.zero_state(self.config.batch_size, tf.float32)

    #Defining Logits
    outputOfRecurrentHiddenLayer, states = rnn.rnn(self.multilayerRNN, inputTensorsAsList, initial_state=self._initial_state)
    outputOfRecurrentHiddenLayer = tf.reshape(tf.concat(1, outputOfRecurrentHiddenLayer), [-1, self.config.hidden_size])
    self._logits = tf.nn.xw_plus_b(outputOfRecurrentHiddenLayer, tf.get_variable("softmax_w", [self.config.hidden_size, self.vocabularySize]), tf.get_variable("softmax_b", [self.vocabularySize]))


    #Define the loss
    loss = seq2seq.sequence_loss_by_example([self._logits], [tf.reshape(self._inputTargetsY, [-1])], [tf.ones([self.config.batch_size * self.config.num_time_steps])], self.vocabularySize)
    self._cost = tf.div(tf.reduce_sum(loss), self.config.batch_size)

    self._predictionSoftmax = tf.nn.softmax(self._logits)

    self._final_state = states[-1]




    if not is_training:
      return

    self.defineTensorGradientDescent()


  def defineTensorGradientDescent(self):
    self._learningRate = tf.Variable(0.0, trainable=False)

    trainingVars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, trainingVars),self.config.max_grad_norm)
    optimizer = tf.train.AdamOptimizer(self.learningRate)
    self._tensorGradientDescentTrainingOperation = optimizer.apply_gradients(zip(grads, trainingVars))


  def defineTensorRNNOperation(self,multiLayerRNNCellArch, inputTensorsAsList, config):
    outputOfRecurrentHiddenLayer, states = rnn.rnn(multiLayerRNNCellArch, inputTensorsAsList, initial_state=self._initial_state)

    outputOfRecurrentHiddenLayer = tf.reshape(tf.concat(1, outputOfRecurrentHiddenLayer), [-1, config.hidden_size])
    logits = tf.nn.xw_plus_b(outputOfRecurrentHiddenLayer, tf.get_variable("softmax_w", [config.hidden_size, self.vocabularySize]), tf.get_variable("softmax_b", [self.vocabularySize]))

    return logits, states


  def prepareInputInBatchedFormWithEmbeddings(self, config):


    with tf.device("/cpu:0"): #Tells Tensorflow what GPU to use specifically
      embedding = tf.get_variable("embedding", [self.vocabularySize, config.embeddingSize])
      embeddingLookedUp = tf.nn.embedding_lookup(embedding, self._inputX)
      inputs = tf.split(1, config.num_time_steps, embeddingLookedUp)
      inputTensorsAsList = [tf.squeeze(input_, [1]) for input_ in inputs]
      return inputTensorsAsList

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
  def tensorGradientDescentTrainingOperation(self):
    return self._tensorGradientDescentTrainingOperation

  @property
  def predictionSoftmax(self):
    return self._predictionSoftmax

  @property
  def logits(self):
    return self._logits



