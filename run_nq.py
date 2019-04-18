# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import json
import tensorflow as tf
from tensorflow import metrics
from tensorflow.python.ops import math_ops

from bert import modeling
from bert.run_squad import create_model
from bert import optimization


flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    "eval_steps", 1000,
    "Number of evaluation steps "
    "Number of evaluation steps")

flags.DEFINE_string(
    "bert_data_dir", None,
    "The output directory where the tf records will be written.")

flags.DEFINE_integer(
    "num_train_steps", 100000,
    "Number of total training steps "
    "Number of total training steps")


def read_candidates(input_path):
  """
  map example_ids -> long answer candidates to map short answers to lng answers.
  :param input_path:
  :return:
  """
  import jsonlines
  candidates = {}
  for _file in input_path:
    with jsonlines.open(_file) as reader:
      for i, example in enumerate(reader):
        candidates[example['example_id']] = example['long_answer_candidates']
  return candidates

NB_EPOCHS = 10000

def input_fn_builder(input_files, seq_length, mode):
  """Creates an `input_fn` closure to be passed to Estimator."""
  tf.logging.info(mode)
  name_to_features = {
      "example_id": tf.FixedLenFeature([], tf.int64),
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "start_bytes": tf.FixedLenFeature([seq_length], tf.int64),
      "end_bytes": tf.FixedLenFeature([seq_length], tf.int64),
  }
  if mode == 'train':
    name_to_features["positions"] = tf.FixedLenFeature([2], tf.int64)
  if mode == 'eval':
    name_to_features["positions"] = tf.FixedLenFeature([10], tf.int64)

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
    if mode is not tf.estimator.ModeKeys.PREDICT:
      dt = dt.shuffle(buffer_size=100)
    if mode == tf.estimator.ModeKeys.TRAIN:
      dt = dt.repeat(NB_EPOCHS)
    dt = dt.batch(batch_size)
    return dt

  return input_fn


def argmax_2d(start_l, end_l):
    """
    argmax over start and end logits
    :param start_l: [ batch_size, seq_length]
    :param end_l: [ batch_size, seq_length]
    :return: score, ix [batch_size, 2]
    """
    # exponentiate to make large negative # -> zero
    start_l = tf.math.exp(start_l)
    end_l = tf.math.exp(end_l)
    start_l = tf.expand_dims(start_l, 1)
    end_l = tf.expand_dims(end_l, -1)
    logits = start_l * end_l
    # mask upper triangle.
    logits = tf.linalg.LinearOperatorLowerTriangular(logits).to_dense()
    flat_logits = tf.reshape(logits, shape=[tf.shape(logits)[0], -1])
    _argmax = tf.cast(tf.argmax(flat_logits, axis=-1), dtype=tf.int32)
    ix = tf.cast(tf.stack([_argmax % tf.shape(logits)[1], _argmax // tf.shape(logits)[2]], axis=-1), dtype=tf.int64)
    return tf.cast(tf.reduce_max(flat_logits, axis=-1), tf.int64), ix


def span_accuracy(predictions, positions, n_way=5):
    """
    Exact span match.
    :param predictions: [batch_size, 2]
    :param positions: [batch_size, 5, 2]
    :return: [batch_size]
    """
    predictions = tf.stack(n_way * [predictions], axis=1)
    _equal = tf.cast(math_ops.equal(predictions, positions), tf.int64)
    is_correct = tf.reduce_any(tf.equal(tf.reduce_sum(_equal, axis=-1), 2), axis=-1)
    return is_correct

def precision_and_recall(accuracy, positions):
  """
  calculates precision and recall.
  :param accuracy:
  :param positions:
  :return:
  """
  _equal = tf.cast(math_ops.equal(positions, 0), tf.int64)
  labels = tf.reduce_any(tf.not_equal(tf.reduce_sum(_equal, axis=-1), 2), -1)
  tp = tf.reduce_sum(
      tf.cast(math_ops.logical_and(math_ops.equal(accuracy, True), math_ops.equal(labels, False)), tf.float64))
  fp = tf.reduce_sum(
      tf.cast(math_ops.logical_and(math_ops.equal(accuracy, False), math_ops.equal(labels, False)), tf.float64))
  fn = tf.reduce_sum(
      tf.cast(math_ops.logical_and(math_ops.equal(accuracy, False), math_ops.equal(labels, True)), tf.float64))
  precision = tf.divide(tp, tf.math.maximum(tp + fp, 1))
  recall = tf.divide(tp, tf.math.maximum(tp + fn, 1))
  precision_and_recall_metrics = {'precision': precision,
                                  'recall': recall,
                                  'tp': tp,
                                  'fp': fp}
  return precision_and_recall_metrics


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """
  Returns `model_fn` closure for Estimator.
  Nearly identical to BERT except that it has eval mode as well.
  """

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for Estimator."""
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    # to convert predictions back to bytes
    start_bytes = features["start_bytes"]
    end_bytes = features["end_bytes"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (start_logits, end_logits) = create_model(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=None,
        segment_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    start_ix = tf.argmax(start_logits, axis=-1)  # [batch_size]
    end_ix = tf.argmax(end_logits, axis=-1)  # [batch_size]
    tf.summary.histogram('start_ix', start_ix)
    tf.summary.histogram('end_ix', end_ix)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint:
      assignment_map, initialized_variable_names = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # span predictions [0, seq_length]
    score, y_pred = argmax_2d(start_logits, end_logits)
    if mode == tf.estimator.ModeKeys.PREDICT:
      """
        Prediction format:
          {'predictions': [
            {
              'example_id': -2226525965842375672,
              'long_answer': {
                'start_byte': 62657, 'end_byte': 64776,
                'start_token': 391, 'end_token': 604
            },
              'long_answer_score': 13.5,
              'short_answers': [
                {'start_byte': 64206, 'end_byte': 64280,
                 'start_token': 555, 'end_token': 560}, ...],
              'short_answers_score': 26.4,
              'yes_no_answer': 'NONE'
            }, ... ]
        }
      """
      y_pred_start = tf.cast(tf.one_hot(y_pred[:, 0], depth=tf.shape(start_bytes)[-1]), dtype=tf.int64)
      y_pred_end = tf.cast(tf.one_hot(y_pred[:, 1], depth=tf.shape(end_bytes)[-1]), dtype=tf.int64)
      start_byte = tf.reduce_sum(start_bytes * y_pred_start, axis=-1)
      end_byte = tf.reduce_sum(end_bytes * y_pred_end, axis=-1)

      predictions = {
        "example_id": features["example_id"],
        "input_ids": input_ids,
        "start_logits": start_logits,
        "end_logits": end_logits,
        "y_pred_start": y_pred_start,
        "y_pred_end": y_pred_end,
        "start_byte": start_byte,
        "end_byte": end_byte,
        "score": score
      }
      output_spec = tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions)
      return output_spec
    seq_length = modeling.get_shape_list(input_ids)[1]
    def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss
    # labels - cast to [batch_size, 2] for train or [batch_size, 5, 2] for eval
    positions = features["positions"]
    if mode == tf.estimator.ModeKeys.TRAIN:
      # loss function
      start_loss = compute_loss(start_logits, positions[:, 0])
      end_loss = compute_loss(end_logits, positions[:, 1])
      total_loss = (start_loss + end_loss) / 2.0
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, False)

      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
      return output_spec

    if mode == tf.estimator.ModeKeys.EVAL:
        positions = tf.reshape(positions, shape=(-1, 5, 2))
        sa = span_accuracy(predictions=y_pred, positions=positions)
        accuracy_op = metrics.mean(sa)
        # precision / recall
        precision_and_recall_metrics = precision_and_recall(sa, positions)
        precision_op = metrics.mean(precision_and_recall_metrics['precision'])
        recall_op = metrics.mean(precision_and_recall_metrics['recall'])
        tp_op = metrics.mean(precision_and_recall_metrics['tp'])
        fp_op = metrics.mean(precision_and_recall_metrics['fp'])
        # loss - this takes the first annotation as ground trugh,
        # which might not be the best way to approximate the eval loss
        _positions = positions[:, 0, :]
        # loss function
        start_loss = compute_loss(start_logits, _positions[:, 0])
        end_loss = compute_loss(end_logits, _positions[:, 1])
        total_loss = (start_loss + end_loss) / 2.0
        return tf.estimator.EstimatorSpec(mode,
                                          loss=total_loss,
                                          eval_metric_ops={'span_accuracy': accuracy_op,
                                                           '_precision': precision_op,
                                                           '_recall': recall_op,
                                                           'tp': tp_op,
                                                           'fp': fp_op})
  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.MakeDirs(FLAGS.bert_data_dir)

  _dev_path = os.path.join(FLAGS.bert_data_dir, 'dev')
  _train_path = os.path.join(FLAGS.bert_data_dir, 'train')
  _predict_path = os.path.join(FLAGS.bert_data_dir, 'predict')

  config = tf.estimator.RunConfig(
      save_checkpoints_steps=FLAGS.save_checkpoints_steps, # this also sets when eval starts
      save_summary_steps=50,
      keep_checkpoint_max=10, #train_and_eval does not save the best models, but the most recent ones.
      model_dir=FLAGS.output_dir
  )

  # log p(t|c) not included for the squad training setup.
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=0,
      use_tpu=False,
      use_one_hot_embeddings=False)

  estimator = tf.estimator.Estimator(
      model_fn=model_fn,
      config=config,
      params={'batch_size': FLAGS.train_batch_size})

  if FLAGS.do_train:
    train_files = [os.path.join(_train_path, _file) for _file in os.listdir(_train_path) if _file.endswith(".tf_record")]
    dev_files = [os.path.join(_dev_path, _file) for _file in os.listdir(_dev_path) if _file.endswith(".tf_record")]
    tf.logging.info("{} files found for training".format(len(train_files)))
    tf.logging.info("{} files found for dev".format(len(dev_files)))
    train_input_fn = input_fn_builder(
      input_files=train_files,
      seq_length=FLAGS.max_seq_length,
      mode='train')
    train_dev_fn = input_fn_builder(
      input_files=dev_files,
      seq_length=FLAGS.max_seq_length,
      mode='eval')

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
    # The evaluate will happen after every checkpoint (save_checkpoints_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_dev_fn,steps=FLAGS.eval_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  #TODO: predict and write predictions.
  if FLAGS.do_predict:
    # TODO: _predict_path
    predict_files = [os.path.join(_dev_path, _file) for _file in os.listdir(_dev_path) if
                     _file.endswith(".tf_record")]
    predict_json_files = [os.path.join(_dev_path, _file) for _file in os.listdir(_dev_path) if _file.endswith(".jsonl")]
    predict_input_fn = input_fn_builder(
      input_files=predict_files,
      seq_length=FLAGS.max_seq_length,
      mode='predict')
    results = []
    for batch_result in estimator.predict(predict_input_fn):
      results.extend(batch_result)
      if len(results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(results)))
    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")
    with tf.gfile.Open(output_prediction_file, "w") as f:
      json.dump(results, f, indent=4)
    # get long candidates to map short answers to long answers.
    candidates = read_candidates(predict_json_files)
    candidates_file = os.path.join(FLAGS.output_dir, "candidates.json")
    with tf.gfile.Open(candidates_file, "w") as f:
      json.dump(candidates, f, indent=4)


if __name__ == "__main__":
  tf.logging.info(FLAGS)
  # flags.mark_flag_as_required("vocab_file")
  # flags.mark_flag_as_required("bert_config_file")
  # flags.mark_flag_as_required("output_dir")
  tf.app.run()