
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

# timestamp = str(int(time.time()))
current_run_out_dir = model_run_outputs

tf.flags.DEFINE_string("model_config", "debugWordToken","A type of model. Possible options are: small, medium, large.")
# tf.flags.DEFINE_string("model_config", "debugCharToken","A type of model. Possible options are: small, medium, large.")

tf.flags.DEFINE_string("data_path", "rnnInputData/DT_849Q.txt", "The path point to the training and testing data")
# tf.flags.DEFINE_string("data_path", "rnnInputData/rnn_input_text.txt", "The path point to the training and testing data")

tf.flags.DEFINE_string("checkpoint_path", os.path.join(current_run_out_dir, "ModelCheckpoint"), "Model Checkpoints")
tf.flags.DEFINE_integer("checkpoint_every", 1, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("print_status_update_every", 100, "Prints a status update of the models after this many steps (default: 100)")

tf.flags.DEFINE_integer("tensorboard_status_update_every", 100, "Records a tensorboard summary of the models after this many steps (default: 100)")
tensorboard_dir = os.path.join(current_run_out_dir, "TensorboardSummary")
tf.flags.DEFINE_string("tensorboard_path", os.path.abspath(tensorboard_dir), "Tensorboard reports")


if not os.path.exists(tf.flags.FLAGS.checkpoint_path):
    os.makedirs(tf.flags.FLAGS.checkpoint_path)

global_step = 0
lowest_validation_perplexity = 1000

def run_epoch(modelType, data_reader, session, model, data, tensorOperationToPerform, consolePrint, summary_writer, saver):

  global global_step, lowest_validation_perplexity

  start_time = time.time()
  accumulatedCosts = 0.0
  accumulatedNumberOfTimeSteps = 0
  currentModelState = model.initial_state.eval()


  #TODO refactor this into a function called single training step

  #TODO I need to refactor this code and run the train and dev within the same x_stepsBatchedInputData blocks like in https://github.com/dennybritz/cnn-text-classification-tf/blob/master/train.py
  # that might solve my problem

  lowest_perplexity = 2000

  for num_time_steps_blocksCounter, (x_stepsBatchedInputData, y_stepsBatchedOutputData) in enumerate(data_reader.generateXYPairIterator(data, model.config.batch_size, model.config.num_time_steps)):
    global_step= global_step+1
    feed_dict = {model._inputX: x_stepsBatchedInputData, model._inputTargetsY: y_stepsBatchedOutputData, model.initial_state: currentModelState}

    cost, currentModelState, summaryOutput, _ = session.run([model.cost,  model.final_state, model.merged_summary_tensorOperation, tensorOperationToPerform], feed_dict)
    accumulatedCosts += cost
    accumulatedNumberOfTimeSteps += model.config.num_time_steps
    perplexity =  np.exp(accumulatedCosts / accumulatedNumberOfTimeSteps)

    speed = accumulatedNumberOfTimeSteps * model.config.batch_size / (time.time() - start_time)

    if modelType == "training" and num_time_steps_blocksCounter != 0 and num_time_steps_blocksCounter % tf.flags.FLAGS.checkpoint_every == 0:
      summary_writer.add_summary(summaryOutput,global_step)
      consolePrint.print_batch_status(model.model_name, num_time_steps_blocksCounter, perplexity, speed)
      if perplexity < lowest_perplexity:
        lowest_perplexity = perplexity
        get_prediction(data_reader, session, 500, get_initial_see_tokens(data_reader.char_mode))


      # global validation_model
      # validation_perplexity, validation_summaryOutput = run_epoch("validating", data_reader, session, validation_model, data_reader.get_validation_data(), tf.no_op(), consolePrint, summary_writer, saver)
      # summary_writer.add_summary(validation_summaryOutput,global_step)
      # if validation_perplexity < lowest_validation_perplexity:
      #   lowest_validation_perplexity = validation_perplexity
      #   consolePrint.print_batch_status(model.model_name, num_time_steps_blocksCounter, perplexity, speed)
      #   model_values = 'epoch_val_perplexity_%.2f_globalStep' % (lowest_validation_perplexity)
      #   print("creating a checkpoint file")
      #   checkpoint_file_path = os.path.join(tf.flags.FLAGS.checkpoint_path, model_values)
      #   saver.save(session, checkpoint_file_path,global_step=global_step)

  return perplexity, summaryOutput


def main(unused_args):

  if not tf.flags.FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  data_reader = DataReader(tf.flags.FLAGS.data_path,False,5,True)
  data_reader.print_data_info()

  consolePrint = ConsolePrint()

  config = hyperParamConfig.get_config(tf.flags.FLAGS.model_config)


  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.variable_scope("model", reuse=None, initializer=initializer):
      training_model = CharRNNModel("Training", data_reader.vocabularySize, is_training=True, config_param=config)
    with tf.variable_scope("model", reuse=True, initializer=initializer):
      global validation_model
      validation_model = CharRNNModel("Validation", data_reader.vocabularySize, is_training=False, config_param=config)

      eval_config = hyperParamConfig.get_config(tf.flags.FLAGS.model_config)
      #We only want to input one token at a time (not as batches) and get out the next token only
      eval_config.batch_size = 1
      eval_config.num_time_steps = 1
      global test_model
      test_model = CharRNNModel("Testing", data_reader.vocabularySize, is_training=False, config_param=eval_config)


    summary_writer = create_tensorboard_variables(session.graph_def, training_model, validation_model)
    saver = tf.train.Saver(tf.all_variables())

    tf.initialize_all_variables().run()

    consolePrint.config_epoch_print_settings(len(data_reader.get_training_data()),training_model.config,10)

    for epochCount in range(config.total_max_epoch):
      consolePrint.epochCount+=1
      learningRateDecay = config.lr_decay ** max(epochCount - config.initialLearningRate_max_epoch, 0.0)
      training_model.assign_learningRate(session, config.learning_rate * learningRateDecay)

      run_epoch("training", data_reader, session, training_model, data_reader.get_training_data(), training_model.tensorGradientDescentTrainingOperation, consolePrint, summary_writer, saver)

    run_epoch("testing", data_reader, session, test_model, data_reader.get_test_data(), tf.no_op(), consolePrint, summary_writer, saver)

  session.close()

def create_tensorboard_variables(session_graph, training_model, validation_model):
  training_logit_summary = tf.histogram_summary("Training Logits", training_model._logits)
  training_learning_rate_summary = tf.scalar_summary("Training Learning Rate", training_model._learningRate)
  training_cost_summary = tf.scalar_summary("Training Cost Summary", training_model._cost)

  training_model.merged_summary_tensorOperation = tf.merge_summary([training_logit_summary, training_learning_rate_summary, training_cost_summary])

  validation_cost_summary = tf.scalar_summary("Validation Cost Summary", validation_model._cost)
  validation_model.merged_summary_tensorOperation = tf.merge_summary([validation_cost_summary])

  summary_writer = tf.train.SummaryWriter(tf.flags.FLAGS.tensorboard_path, session_graph)

  return summary_writer

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
  if dataReader.char_mode ==True:
    for token in output_tokens:
      output_sentence+=token
  else:
    output_sentence = output_sentence.join(str(token) for token in output_tokens)
  print('---- Prediction: \n %s \n----' % (output_sentence))


def get_initial_see_tokens(char_mode):
  if char_mode ==True:
    return ['T','h','e',' ']
  elif tf.flags.FLAGS.data_path == 'rnnInputData/DT_849Q.txt':
    list_of_DT_topics = ['Jeb','Ben','Hillary','Ted','Carly','Lindsey','John','Martin','George','Rick','Marco','Bernie']
    return [np.random.choice(list_of_DT_topics)]
  return [' ']


if __name__ == "__main__":
  tf.app.run()
