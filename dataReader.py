# pylint: disable=unused-import,g-bad-import-order

"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import time

import tensorflow.python.platform

import numpy as np
# from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

class DataReader(object):

  def __init__(self, data_path, validation_data_to_training_ratio=5):

    self.raw_data = open(data_path, 'r').read()

    self.validation_data_to_training_ratio = validation_data_to_training_ratio
    self._build_vocab_dict()
    self._convert_raw_data_to_token_ids()
    self._create_data_indexes()


  def print_data_info(self):
    print('----------------------------------------')
    print('Data total tokens: %d tokens' % (len(self.raw_data)))
    print('Data vocabulary size: %d tokens' % (len(self.unique_tokens)))
    print('Training Data total tokens: %d tokens' % (len(self.get_training_data())))
    print('Validation Data total tokens: %d tokens' % (len(self.get_validation_data())))
    print('Test Data total tokens: %d tokens' % (len(self.get_test_data())))
    print('----------------------------------------')

  def _read_words(self, filename):
    with gfile.GFile(filename, "r") as f:
      return f.read().replace("\n", " <eos> ").split()

  def _build_vocab_dict(self):
    counter = collections.Counter(self.raw_data)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    self.unique_tokens, _ = list(zip(*count_pairs))
    self.token_to_id = dict(zip(self.unique_tokens, range(len(self.unique_tokens))))
    self.vocabularySize = len(self.unique_tokens)


  def convertDigitsToText(self, token_predicted_per_batch,  first_token=" "):

    for token_predicted_per_batch_item in token_predicted_per_batch:
      first_token.join(self.id_to_token[token_predicted_per_batch_item] )

    return first_token


  def _create_data_indexes(self):
    data_size = len(self.data_as_ids)
    training_data_ratio = 100 - (self.validation_data_to_training_ratio*2)
    self.train_data_max_index = (data_size * training_data_ratio)//100
    self.valid_data_min_index = self.train_data_max_index+1
    self.valid_data_max_index = ((data_size * self.validation_data_to_training_ratio)//100)+self.train_data_max_index
    self.test_data_min_index = self.valid_data_max_index+1

  def _convert_raw_data_to_token_ids(self):
    print('Converting %d tokens into ids' % (len(self.raw_data)))
    self.data_as_ids = []
    for id, token in enumerate(self.raw_data):
      epochPercentageAccomplished = id * 100.0 / len(self.raw_data)
      if epochPercentageAccomplished%0.5 ==0:
        print("%.3f%% percent tokens converted" % (epochPercentageAccomplished))
      if token in self.token_to_id.keys():
        self.data_as_ids.append(self.token_to_id[token])
      else:
        print("WARNING: A TOKEN COULD NOT BE ASSIGNED AN ID. SOMETHING MUST HAVE GONE WRONG IN THE INITIAL TOKEN TO ID MAPPING")
        #TODO this should not be there, I'm currently doing it to avoid breaking whenever a char with no id is found instead I should be replacing the entire method with the following line: # return [word_to_id[word] for word in data]
        self.data_as_ids.append(0)



  def get_training_data(self):
    return self.data_as_ids[0:self.train_data_max_index]

  def get_validation_data(self):
    return self.data_as_ids[self.valid_data_min_index:self.valid_data_max_index]

  def get_test_data(self):
    #I don't get what's going on there
    length = len(self.data_as_ids)-1
    return self.data_as_ids[self.test_data_min_index:length]

  def generateXYPairIterator(self, raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
      data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
      raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
      x = data[:, i*num_steps:(i+1)*num_steps]
      y = data[:, i*num_steps+1:(i+1)*num_steps+1]
      yield (x, y)



  def limit_data_size(self, original_data, max_size=None):
    if max_size==None:
      return original_data

    print("Limiting Input data size to:  %d tokens" % (max_size))
    return original_data[0:max_size]
