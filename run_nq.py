# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf

from bert import modeling
from bert.run_squad import create_model
from bert import optimization

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "nb_examples", None,
    "Number of examples extracted. "
    "Determines learning rate bc BERT uses Adam decay optimizer")

flags.DEFINE_string(
    "eval_after_steps", None,
    "Number of steps of training before eval."
    "")


NB_EPOCHS = 1


def input_fn_builder(input_files, seq_length, is_training, mode):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
      "unique_ids": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "start_bytes": tf.FixedLenFeature([seq_length], tf.int64),
      "end_bytes": tf.FixedLenFeature([seq_length], tf.int64),
  }
  if is_training:
    name_to_features["start_positions"] = tf.FixedLenFeature([], tf.int64)
    name_to_features["end_positions"] = tf.FixedLenFeature([], tf.int64)

  def _decode_record(record):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)
    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    dt = tf.data.TFRecordDataset(input_files)
    dt = dt.map(_decode_record, num_parallel_calls=10)
    if is_training:
      if mode == tf.estimator.ModeKeys.TRAIN:
        # do not shuffle and repeat if eval.
        dt = dt.shuffle(buffer_size=100)
        dt = dt.repeat(NB_EPOCHS)
    dt = dt.batch(batch_size)
    return dt

  return input_fn


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """
  Returns `model_fn` closure for Estimator.
  Nearly identical to BERT except that it has eval mode as well.
  """

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    unique_ids = features["unique_ids"]
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    # convert predictions back to bytes
    start_bytes = features["start_bytes"]
    end_bytes = features["end_bytes"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.PREDICT:
      predictions = {
        "unique_ids": unique_ids,
        "start_logits": start_logits,
        "end_logits": end_logits,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      return output_spec
    seq_length = modeling.get_shape_list(input_ids)[1]
    def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

    start_positions = features["start_positions"] #[batch_size]
    end_positions = features["end_positions"] #[batch_size]

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)

    total_loss = (start_loss + end_loss) / 2.0
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
      return output_spec
    if mode == tf.estimator.ModeKeys.EVAL:
        # TODO: Precision / recall / accuracy
        start_ix = tf.argmax(start_logits, axis=-1)
        end_ix = tf.argmax(end_logits, axis=-1)
        acc = tf.metrics.accuracy(labels=start_positions, predictions=start_ix)
        acc = tf.metrics.accuracy(labels=end_positions, predictions=end_ix)





        metrics = None
        return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)
  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  #TODO: Where do I get this from?
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  _dev_path = os.path.join(FLAGS.bert_data_dir, 'dev')
  _train_path = os.path.join(FLAGS.bert_data_dir, 'train')

  config = tf.estimator.RunConfig(
      save_checkpoints_steps=100,
      save_summary_steps=50,
      keep_checkpoint_max=2,
      model_dir=FLAGS.output_dir
  )

  num_warmup_steps = int(FLAGS.num_train_steps * FLAGS.warmup_proportion)
  # log p(t|c) not included for the squad training setup.
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_files = [os.path.join(_train_path, _file) for _file in os.listdir(_train_path) if _file.endswith(".tf_record")]
    dev_files = [os.path.join(_dev_path, _file) for _file in os.listdir(_dev_path) if _file.endswith(".tf_record")]
    train_input_fn = input_fn_builder(
      input_files=train_files,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      mode='train')
    train_dev_fn = input_fn_builder(
      input_files=dev_files,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      mode='eval')

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=FLAGS.eval_after_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_dev_fn)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  #TODO: predict and write predictions.

  if FLAGS.do_predict:
    raise ValueError("Not implemented..")

if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()