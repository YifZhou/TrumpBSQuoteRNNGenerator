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
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.platform import gfile

class ConsolePrint(object):
  def __init__(self):
    self.epochCount = 0

  def config_epoch_print_settings(self, data_len, modelConfig, status_gaps):
    self.totalTimeStepsBlockInEpoch = (  (data_len // modelConfig.batch_size) - 1) // modelConfig.num_time_steps
    self.status_gaps = status_gaps


  def print_final_epoch_summary(self, learningRate, train_perplexity):
    print("Epoch: %d Learning rate: %.3f" % (self.epochCount + 1, learningRate))
    print("Epoch: %d Train Perplexity: %.3f" % (self.epochCount + 1, train_perplexity))

  def print_batch_status(self,model_name, num_time_steps_blocksCounter, perplexity, speed):
    epochPercentageAccomplished = num_time_steps_blocksCounter * 100.0 / self.totalTimeStepsBlockInEpoch
    print("Model: "+model_name+" , Epoch %d %.3f%%, Perplexity: %.3f , Speed: %.0f wps" % (self.epochCount, epochPercentageAccomplished, perplexity, speed))


