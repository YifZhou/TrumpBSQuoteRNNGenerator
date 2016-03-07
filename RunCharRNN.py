
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from rnnModel import CharRNNModel
import time
import os
from BSReader import DataReader
import numpy as np
import shutil
import tensorflow as tf


logging = tf.logging

model_run_outputs = os.path.join(os.path.curdir, "ModelRunsOutput/CurrentRuns")
if os.path.exists(model_run_outputs):
  shutil.rmtree(model_run_outputs)


current_run_out_dir = model_run_outputs

tf.flags.DEFINE_string("model_config", "debugCharToken","A type of model. Possible options are: small, medium, large.")

tf.flags.DEFINE_string("data_path", "TrumpBSQuotes.txt", "The path point to the training and testing data")

tf.flags.DEFINE_string("checkpoint_path", os.path.join(current_run_out_dir, "ModelCheckpoint"), "Model Checkpoints")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("print_status_update_every", 100, "Prints a status update of the models after this many steps (default: 100)")



if not os.path.exists(tf.flags.FLAGS.checkpoint_path):
    os.makedirs(tf.flags.FLAGS.checkpoint_path)


epochCount = 0
lowest_validation_perplexity = 1000


def main(unused_args):

  if not tf.flags.FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  data_reader = DataReader(tf.flags.FLAGS.data_path,5)
  data_reader.print_data_info()


  config = HyperParameterConfig()


  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.variable_scope("model", reuse=None, initializer=initializer):
      training_model = CharRNNModel(data_reader.vocabularySize, is_training=True, config_param=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      eval_config = HyperParameterConfig()
      #We only want to input one token at a time (not as batches) and get out the next token only
      eval_config.batch_size = 1
      eval_config.num_time_steps = 1
      global test_model
      test_model = CharRNNModel(data_reader.vocabularySize, is_training=False, config_param=eval_config)

    tf.initialize_all_variables().run()



    for epochCount in range(config.total_max_epoch):
      epochCount+=1
      learningRateDecay = config.lr_decay ** max(epochCount - config.initialLearningRate_max_epoch, 0.0)
      training_model.assign_learningRate(session, config.learning_rate * learningRateDecay)

      global lowest_validation_perplexity

      accumulatedCosts = 0.0
      accumulatedNumberOfTimeSteps = 0
      currentModelState = training_model.initial_state.eval()


      lowest_perplexity = 2000

      for num_time_steps_blocksCounter, (x_stepsBatchedInputData, y_stepsBatchedOutputData) in enumerate(data_reader.generateXYPairIterator(data_reader.get_training_data(), training_model.config.batch_size, training_model.config.sequence_size)):

        feed_dict = {training_model._inputX: x_stepsBatchedInputData, training_model._inputTargetsY: y_stepsBatchedOutputData, training_model.initial_state: currentModelState}

        cost, currentModelState, _ = session.run([training_model.cost,  training_model.final_state, training_model.tensorGradientDescentTrainingOperation], feed_dict)
        accumulatedCosts += cost
        accumulatedNumberOfTimeSteps += training_model.config.sequence_size
        perplexity =  np.exp(accumulatedCosts / accumulatedNumberOfTimeSteps)


        if  num_time_steps_blocksCounter != 0 and num_time_steps_blocksCounter % tf.flags.FLAGS.checkpoint_every == 0:
          epochPercentageAccomplished = num_time_steps_blocksCounter * 100.0 / ((  (len(data_reader.get_training_data()) // training_model.config.batch_size) - 1) // training_model.config.sequence_size)
          print("Epoch %d %.3f%%, Perplexity: %.3f" % (epochCount, epochPercentageAccomplished, perplexity))

          if perplexity < lowest_perplexity:
            lowest_perplexity = perplexity
            get_prediction(data_reader, session, 500, ['T','h','e',' '])

  session.close()



def get_prediction(dataReader, session, total_tokens, output_tokens = [' ']):
  global test_model

  state = test_model.multilayerRNN.zero_state(1, tf.float32).eval()

  for token_count in xrange(total_tokens):
      next_token = output_tokens[token_count]
      input = np.full((test_model.config.batch_size, test_model.config.sequence_size), dataReader.token_to_id[next_token], dtype=np.int32)
      feed = {test_model._inputX: input, test_model._initial_state:state}
      [predictionSoftmax, state] =  session.run([test_model._predictionSoftmax, test_model._final_state], feed)

      if (len(output_tokens) -1) <= token_count:
          accumulated_sum = np.cumsum(predictionSoftmax[0])
          currentTokenId = (int(np.searchsorted(accumulated_sum, np.random.rand(1))))
          next_token = dataReader.unique_tokens[currentTokenId]
          output_tokens.append(next_token)

  output_sentence = " "

  for token in output_tokens:
    output_sentence+=token
  print('---- Prediction: \n %s \n----' % (output_sentence))


class HyperParameterConfig(object):
  init_scale = 0.1
  learning_rate = 0.002
  max_grad_norm = 5
  num_layers = 2
  sequence_size = 50
  batch_size = 50
  hidden_size = 128
  embeddingSize = 100
  initialLearningRate_max_epoch = 1
  total_max_epoch = 10000
  keep_prob = 1.0
  lr_decay = 0.97



if __name__ == "__main__":
  tf.app.run()

