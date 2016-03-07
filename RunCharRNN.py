
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hyperParamConfig
from rnnModel import CharRNNModel
import time
import os
from consolePrint import ConsolePrint
from dataReader import DataReader
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

global_step = 0
lowest_validation_perplexity = 1000

def run_epoch(modelType, data_reader, session, model, data, tensorOperationToPerform, consolePrint):

  global global_step, lowest_validation_perplexity

  start_time = time.time()
  accumulatedCosts = 0.0
  accumulatedNumberOfTimeSteps = 0
  currentModelState = model.initial_state.eval()


  lowest_perplexity = 2000

  for num_time_steps_blocksCounter, (x_stepsBatchedInputData, y_stepsBatchedOutputData) in enumerate(data_reader.generateXYPairIterator(data, model.config.batch_size, model.config.num_time_steps)):
    global_step= global_step+1
    feed_dict = {model._inputX: x_stepsBatchedInputData, model._inputTargetsY: y_stepsBatchedOutputData, model.initial_state: currentModelState}

    cost, currentModelState, _ = session.run([model.cost,  model.final_state, tensorOperationToPerform], feed_dict)
    accumulatedCosts += cost
    accumulatedNumberOfTimeSteps += model.config.num_time_steps
    perplexity =  np.exp(accumulatedCosts / accumulatedNumberOfTimeSteps)

    speed = accumulatedNumberOfTimeSteps * model.config.batch_size / (time.time() - start_time)

    if modelType == "training" and num_time_steps_blocksCounter != 0 and num_time_steps_blocksCounter % tf.flags.FLAGS.checkpoint_every == 0:
      consolePrint.print_batch_status(model.model_name, num_time_steps_blocksCounter, perplexity, speed)
      if perplexity < lowest_perplexity:
        lowest_perplexity = perplexity
        get_prediction(data_reader, session, 500, ['T','h','e',' '])


  return perplexity


def main(unused_args):

  if not tf.flags.FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  data_reader = DataReader(tf.flags.FLAGS.data_path,5)
  data_reader.print_data_info()

  consolePrint = ConsolePrint()

  config = hyperParamConfig.get_config()


  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.variable_scope("model", reuse=None, initializer=initializer):
      training_model = CharRNNModel("Training", data_reader.vocabularySize, is_training=True, config_param=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      eval_config = hyperParamConfig.get_config()
      #We only want to input one token at a time (not as batches) and get out the next token only
      eval_config.batch_size = 1
      eval_config.num_time_steps = 1
      global test_model
      test_model = CharRNNModel("Testing", data_reader.vocabularySize, is_training=False, config_param=eval_config)

    tf.initialize_all_variables().run()

    consolePrint.config_epoch_print_settings(len(data_reader.get_training_data()),training_model.config,10)

    for epochCount in range(config.total_max_epoch):
      consolePrint.epochCount+=1
      learningRateDecay = config.lr_decay ** max(epochCount - config.initialLearningRate_max_epoch, 0.0)
      training_model.assign_learningRate(session, config.learning_rate * learningRateDecay)

      run_epoch("training", data_reader, session, training_model, data_reader.get_training_data(), training_model.tensorGradientDescentTrainingOperation, consolePrint)


  session.close()



def get_prediction(dataReader, session, total_tokens, output_tokens = [' ']):
  global test_model

  state = test_model.cell.zero_state(1, tf.float32).eval()

  for token_count in xrange(total_tokens):
      next_token = output_tokens[token_count]
      input = np.full((test_model.config.batch_size, test_model.config.num_time_steps), dataReader.token_to_id[next_token], dtype=np.int32)
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




if __name__ == "__main__":
  tf.app.run()
