# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import math
import os
import random
import six
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }
  if is_training:
    name_to_features["start_position"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_position"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.map(_decode_record, num_parallel_calls=10)
      d = d.repeat()
      d = d.batch(batch_size)
      d = d.shuffle(buffer_size=100)
    return d

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure"""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) =  xsc (
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    if mode == tf.estimator.ModeKeys.TRAIN:
      seq_length = modeling.get_shape_list(input_ids)[1]
      s = features["start_position"]
      e = features["end_position"]


  #TODO: estimator spec return

  return model_fn