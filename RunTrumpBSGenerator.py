
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from RNN_Model import TrumpBSModel
from BSReader import BSReader
import numpy as np
import tensorflow as tf


tf.flags.DEFINE_string("dp", "TrumpBSQuotes.txt", "The path point to the training and testing data")
tf.flags.DEFINE_integer("ckpt", 1, "Checkpoint after this many steps (default: 100)")

def main(unused_args):

  if not tf.flags.FLAGS.dp:
    raise ValueError("Must set --data_path to PTB data directory")

  bs_reader = BSReader(tf.flags.FLAGS.dp,5)
  bs_reader.print_data_info()


  with tf.Graph().as_default(), tf.Session() as session:
    config = HyperParameterConfig()
    initializer = tf.random_uniform_initializer(-config.init_scale,config.init_scale)

    with tf.variable_scope("TrumpBSQuoteModel", reuse=None, initializer=initializer):
      training_model = TrumpBSModel(bs_reader.vocabularySize,config_param=config)
      training_model.defineTensorGradientDescent()
    with tf.variable_scope("TrumpBSQuoteModel", reuse=True, initializer=initializer):
      eval_config = HyperParameterConfig()
      #We only want to input one token at a time (not as batches) and get out the next token only
      eval_config.batch_size = 1
      eval_config.num_time_steps = 1
      prediction_model = TrumpBSModel(bs_reader.vocabularySize, config_param=eval_config)

      tf.initialize_all_variables().run()

    for epochCount in range(config.total_max_epoch):

      accumulated_costs = 0.0
      accumulated_seq_count = 0
      current_model_state = training_model.initial_state.eval()

      #This can be removed and replaced by assigning just the initial learning rate
      learning_rate_decay = config.lr_decay ** max(epochCount - config.initialLearningRate_max_epoch, 0.0)
      training_model.assign_learningRate(session, config.learning_rate * learning_rate_decay)

      lowest_perplexity = 2000

      for sequence_counter, (x, y) in enumerate(bs_reader.generateXYPairs(bs_reader.get_training_data(), training_model.config.batch_size, training_model.config.sequence_size)):

        feed_dict = {training_model._inputX: x, training_model._inputTargetsY: y, training_model.initial_state: current_model_state}

        cost, current_model_state, _ = session.run([training_model.cost, training_model.final_state, training_model.gradient_desc_training_op], feed_dict)
        accumulated_costs += cost
        accumulated_seq_count += training_model.config.sequence_size
        perplexity =  np.exp(accumulated_costs / accumulated_seq_count)

        if  sequence_counter != 0 and sequence_counter % tf.flags.FLAGS.ckpt == 0:
          print("Epoch %d, Perplexity: %.3f" % (epochCount, perplexity))

          if perplexity < lowest_perplexity:
            lowest_perplexity = perplexity
            get_prediction(prediction_model, bs_reader, session, 500, ['T','h','e',' '])

  session.close()



def get_prediction(model, bs_Reader, session, total_tokens, output_tokens = [' ']):

  state = model.multilayerRNN.zero_state(1, tf.float32).eval()

  for token_count in range(total_tokens):
      next_token = output_tokens[token_count]
      input = np.full((model.config.batch_size, model.config.sequence_size), bs_Reader.token_to_id[next_token], dtype=np.int32)
      feed = {model._inputX: input, model._initial_state:state}
      [predictionSoftmax, state] =  session.run([model._predictionSoftmax, model._final_state], feed)

      if (len(output_tokens) -1) <= token_count:
          accumulated_sum = np.cumsum(predictionSoftmax[0])
          currentTokenId = (int(np.searchsorted(accumulated_sum, np.random.rand(1))))
          next_token = bs_Reader.unique_tokens[currentTokenId]
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

