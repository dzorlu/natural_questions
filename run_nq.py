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

from bert import modeling
from bert.run_squad import model_fn_builder

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "nb_examples", None,
    "Number of examples extracted. "
    "Determines learning rate bc BERT uses Adam decay optimizer")


def input_fn_builder(input_file, seq_length, is_training):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }
  if is_training:
    name_to_features["start_position"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_position"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    dt = tf.data.TFRecordDataset(input_file)
    if is_training:
      dt = dt.map(_decode_record, num_parallel_calls=10)
      dt = dt.repeat()
      dt = dt.batch(batch_size)
      dt = dt.shuffle(buffer_size=100)
    return dt

  return input_fn



def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  #TODO: Break it into train / predict
  _dev_path = os.path.join(FLAGS.bert_data_dir, 'dev')
  _train_path = os.path.join(FLAGS.bert_data_dir, 'train')
  train_files = [_file for _file in os.listdir(_train_path) if _file.endswith(".tf_record")]
  _file_path = [os.path.join(_train_path, _file) for _file in train_files]
  train_input_fn = input_fn_builder(
    input_file=_file_path,
    seq_length=FLAGS.max_seq_length,
    is_training=True)

  num_warmup_steps = int(FLAGS.num_train_steps * FLAGS.warmup_proportion)
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)



if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()