# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import tensorflow as tf
from tensorflow import metrics
from tensorflow.python.ops import math_ops
from tensorflow.train import AdamOptimizer

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



NB_EPOCHS = 10000

def input_fn_builder(input_files, seq_length, is_training, mode):
  """Creates an `input_fn` closure to be passed to Estimator."""

  name_to_features = {
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
      # shuffles for eval
      dt = dt.shuffle(buffer_size=100)
      if mode == tf.estimator.ModeKeys.TRAIN:
        # do not shuffle and repeat if eval.
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

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)
      tf.summary.histogram(var.name, var)
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

      # TODO: This needs to argmax on two dims.
      start_ix = tf.argmax(start_logits, axis=-1) # [batch_size]
      end_ix = tf.argmax(end_logits, axis=-1)  # [batch_size]
      start_bytes_ix = tf.batch_gather(start_bytes, start_ix)
      end_bytes_ix = tf.batch_gather(end_bytes, end_ix)


      predictions = {
        "input_ids": input_ids,
        "start_bytes_ix": start_bytes_ix,
        "end_bytes_ix": end_bytes_ix,
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
    # labels
    start_positions = features["start_positions"] #[batch_size]
    end_positions = features["end_positions"] #[batch_size]

    # loss function
    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)
    total_loss = (start_loss + end_loss) / 2.0

    def argmax_2d(start_l, end_l):
      """
      argmax over start and end logits
      :param start_l:
      :param end_l:
      :return:
      """
      start_l = tf.expand_dims(start_l, 1)
      end_l = tf.expand_dims(end_l, -1)
      logits = start_l * end_l
      flat_logits = tf.reshape(logits, shape=[tf.shape(logits)[0], -1])
      _argmax = tf.cast(tf.argmax(flat_logits, axis=-1), dtype=tf.int32)
      return tf.cast(tf.stack([_argmax % tf.shape(logits)[1], _argmax // tf.shape(logits)[2]], axis=-1), dtype=tf.int64)

    if mode == tf.estimator.ModeKeys.TRAIN:
      # debugging the optimizer
      step = tf.train.get_global_step()
      opt = AdamOptimizer(learning_rate=learning_rate)
      gradient_variable = opt.compute_gradients(total_loss)
      for g,v in gradient_variable:
        if g is not None:
          tf.summary.histogram("%s-grad" % v.name, g)
      train_op = opt.apply_gradients(gradient_variable, global_step=step)
      # debug end
      # train_op = optimization.create_optimizer(
      #     total_loss, learning_rate, num_train_steps, num_warmup_steps, False)
      output_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op)
      return output_spec

    if mode == tf.estimator.ModeKeys.EVAL:
        # # TODO: Precision / recall
        def span_accuracy(start_logits, end_logits, start_positions, end_positions):
            """
            Exact span match.
                pred -> [10,20] gt -> [10,20] : True
                pred -> [10,20] gt -> [10,15] : False
            :param start_logits: [batch_size, seq_lenght]
            :param end_logits: [batch_size, seq_lenght]
            :param start_positions: [batch_size]
            :param end_positions: [batch_size]
            :return:   
                accuracy: A `Tensor` representing the accuracy, the value of `total` divided
                  by `count`.
                update_op: An operation that increments the `total` and `count` variables
                  appropriately and whose value matches `accuracy`.
            """

            y_pred_ix = argmax_2d(start_logits, end_logits)
            start_positions = tf.expand_dims(start_positions, axis=-1)
            end_positions = tf.expand_dims(end_positions, axis=-1)
            y_true_ix = tf.concat([start_positions, end_positions], axis=-1) #[batch_size, 2]
            acc = tf.reduce_all(math_ops.equal(y_true_ix, y_pred_ix), axis=-1)
            is_correct = math_ops.to_float(acc)
            return metrics.mean(is_correct)

        sa = span_accuracy(start_logits, end_logits, start_positions, end_positions)
        # add it to tensorboard
        tf.summary.scalar('accuracy', sa[1])
        return tf.estimator.EstimatorSpec(mode,
                                          loss=total_loss,
                                          eval_metric_ops={'span_accuracy': sa})
    return tf.estimator.EstimatorSpec(mode, loss=total_loss)
  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
  tf.gfile.MakeDirs(FLAGS.output_dir)
  tf.gfile.MakeDirs(FLAGS.bert_data_dir)

  _dev_path = os.path.join(FLAGS.bert_data_dir, 'dev')
  _train_path = os.path.join(FLAGS.bert_data_dir, 'train')

  config = tf.estimator.RunConfig(
      save_checkpoints_steps=FLAGS.save_checkpoints_steps, # this also sets when eval starts
      save_summary_steps=50,
      keep_checkpoint_max=10, #train_and_eval does not save the best models, but the most recent ones.
      model_dir=FLAGS.output_dir
  )

  #num_warmup_steps = int(FLAGS.num_train_steps * 0.01)
  #tf.logging.info("nb training steps: {}".format(num_warmup_steps))
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
      is_training=True,
      mode='train')
    train_dev_fn = input_fn_builder(
      input_files=dev_files,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      mode='eval')

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)
    # The evaluate will happen after every checkpoint (save_checkpoints_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=train_dev_fn,steps=FLAGS.eval_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

  #TODO: predict and write predictions.
  if FLAGS.do_predict:
    raise ValueError("Not implemented..")

if __name__ == "__main__":
  tf.logging.info(FLAGS)
  # flags.mark_flag_as_required("vocab_file")
  # flags.mark_flag_as_required("bert_config_file")
  # flags.mark_flag_as_required("output_dir")
  tf.app.run()