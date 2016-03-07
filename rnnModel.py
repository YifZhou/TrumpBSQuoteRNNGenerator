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
    print('Building model: %s ' %(model_name))
    self.model_name = model_name
    self.vocabularySize = vocabularySize
    self.config = config_param
    self._inputX = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_time_steps], "InputsX")
    self._inputTargetsY = tf.placeholder(tf.int32, [self.config.batch_size, self.config.num_time_steps], "InputTargetsY")

    multiLayerRNNCellArch =  self.defineTensorRNNArchitecture(self.config.hidden_size, self.config.num_layers, self.config.batch_size)

    self._initial_state = multiLayerRNNCellArch.zero_state(self.config.batch_size, tf.float32)

    inputTensorsAsList = self.prepareInputInBatchedFormWithEmbeddings(self.config)

    #logit refers to the logit function (inverse of the sigmoidal logistic function ) https://en.wikipedia.org/wiki/Logit the logit or logistical function is used to convert values into probabilities
    #logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols] [batch_size x vocabularySize]
      # This is confusing because it's basically the batches put back in a list so technically it's not batch_size BUT (batch_size multipliedBy num_steps x vocabularySize)
    #So it's the array of probabilities of what character to output PER batch though!
    self._logits, states = self.defineTensorRNNOperation(multiLayerRNNCellArch, inputTensorsAsList, self.config)


    self.defineTensorLoss(self._logits)

    self._predictionSoftmax = tf.nn.softmax(self._logits)

    # self._charactersPredictedPerBatch = tf.arg_max(logits, 1)

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

  def defineTensorLoss(self,logits):
    loss = seq2seq.sequence_loss_by_example([logits], [tf.reshape(self._inputTargetsY, [-1])], [tf.ones([self.config.batch_size * self.config.num_time_steps])], self.vocabularySize)
    #Killian: apparently the following line below was a bug so replacing it as suggested in http://stackoverflow.com/questions/34010962/tesnor-flow-unsupported-opperand
    #  self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._cost = tf.div(tf.reduce_sum(loss), self.config.batch_size)



  def defineTensorRNNOperation(self,multiLayerRNNCellArch, inputTensorsAsList, config):
    outputOfRecurrentHiddenLayer, states = rnn.rnn(multiLayerRNNCellArch, inputTensorsAsList, initial_state=self._initial_state)

    outputOfRecurrentHiddenLayer = tf.reshape(tf.concat(1, outputOfRecurrentHiddenLayer), [-1, config.hidden_size])
    logits = tf.nn.xw_plus_b(outputOfRecurrentHiddenLayer, tf.get_variable("softmax_w", [config.hidden_size, self.vocabularySize]), tf.get_variable("softmax_b", [self.vocabularySize]))

    return logits, states


  def defineTensorRNNArchitecture(self, hidden_size, num_layers, batch_size):
    singleRNNCellArch = rnn_cell.BasicRNNCell(hidden_size)
    # if is_training and config.keep_prob < 1:
      # rnnCell = rnn_cell.DropoutWrapper(rnnCell, output_keep_prob=config.keep_prob)
    self.cell =  rnn_cell.MultiRNNCell([singleRNNCellArch] * num_layers)
    return self.cell



  def prepareInputInBatchedFormWithEmbeddings(self, config):


    with tf.device("/cpu:0"): #Tells Tensorflow what GPU to use specifically

      #Killian: this initialises the embeddings to default values (if you want other default values you can specify it as additional params)
      embedding = tf.get_variable("embedding", [self.vocabularySize, config.embeddingSize])

      # returns a tensor of type [batch_size, num_time_steps] - if I understand correctly each element in input_data is converted to a [vocab_size, hiddenSize] tensor
      # so we end up having a [batch_size, num_time_steps] matrix with each element [vocab_size, hiddenSize] [batch_size, num_time_steps, vocab_size, hiddenSize]
      #Killian: this basically simply replaces input_data (containing list of single numbers) with a list of vectors for each number

      #[batch_size x num_time_steps  x embeddingSize] - think about it as embeddingSize being the 3 RGB colors for each pixel (here each pixel is a word) so you end up with a cube representation
      # of an image and 3 layers behind it of R, G, B values, where each pixel has 3 dimensions, here each word has embeddingSize dimensions
      embeddingLookedUp = tf.nn.embedding_lookup(embedding, self._inputX)

      # Creates num_time_steps arrays of shape [batch_size x 1 x  vocab_size x embeddingSize]
      inputs = tf.split(1, config.num_time_steps, embeddingLookedUp)

      #This only squeezes the dimention 1 (column) so does that mean the output is of size vocab_size?
      # Produces num_time_steps arrays each of shape [batch_size x  vocab_size x embeddingSize]
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



