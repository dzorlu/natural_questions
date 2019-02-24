import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
import tensorflow as tf
import tensorflow_hub as hub
import collections
import numpy as np
import multiprocessing
import os
import hashlib

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
CLS = "[CLS]"
SEP = "[SEP]"

DOWNSAMPLE = 50


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer(
    "max_query_length", 64,
    "The maximum number of tokens for the question. Questions longer than "
    "this will be truncated to this length.")


flags.DEFINE_string(
    "predict_file", None,
    "prediction files")

flags.DEFINE_bool("is_training", False, "Whether to run training.")


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["start_position"] = create_int_feature([feature.targets[0]])
      features["end_position"] = create_int_feature([feature.targets[1]])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 targets,
                 tokens,
                 answer_id=1):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.targets = targets
        self.tokens = tokens
        self.answer_id = answer_id

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        if self.targets:
            s += "targets: {}".format(self.targets)
        if self.input_ids:
            s += ", input_ids: {}".format(self.input_ids)
        if self.input_mask:
            s += ", input_mask: {}".format(self.input_mask)
        if self.segment_ids:
            s += ", segment_ids: {}".format(self.segment_ids)
        if self.tokens:
            s += ", tokens: {}".format(self.tokens)
        if self.answer_id:
            s += ", answer_id: {}".format(self.answer_id)
        return s


def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        with tf.Session() as sess:
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                  tokenization_info["do_lower_case"]])

    return bert.tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=do_lower_case)


def convert_example(example,
                    tokenizer,
                    max_seq_length,
                    max_query_length=None,
                    is_training=True,
                    downsample_null_instances=True,
                    train_writer=None):
    """

    :param example: example with following attributes: 
        ['annotations', 'document_html', 'document_title', 'document_tokens', 'document_url', 'example_id', 'long_answer_candidates', 'question_text', 'question_tokens']
    :param tokenizer: 
    :param max_seq_length: sequence length for the model.
    :param is_training: sequence length for the model.
    :param downsample_null_instances
    :param train_writer
    :return: 
    """
    outputs = []
    if not max_query_length:
        max_query_length = max_seq_length // 50
    query_tokens = tokenizer.tokenize(example.get('question_text'))
    if len(query_tokens) > max_query_length:
        query_tokens = query_tokens[0:max_query_length]

    all_doc_tokens = []
    start_bytes = []  # flattened start_byte by sub_token
    end_bytes = []  # flattened end_byte by sub_token
    token_ix = []  # token ix flattened by sub token

    _TargetByteRange = collections.namedtuple(  # pylint: disable=invalid-name
        "TargetByteRange", ["short_start", "short_end", "long_start", "long_end"])

    if is_training:
        # if all annotated short spans are contained in the instance,
        # we set the start and end target indices to point to the
        # smallest span containing all annotated short answer spans
        annotations = next(iter(example.get('annotations')))
        # short answers
        short_answers = annotations.get('short_answers')
        short_answer_byte_start_ix, short_answer_byte_end_ix = -1, -1
        if short_answers:
            short_answer_byte_start_ix = min([sa.get('start_byte') for sa in short_answers])
            short_answer_byte_end_ix = max([sa.get('end_byte') for sa in short_answers])
        # long answers
        long_answer = annotations.get('long_answer')
        # if no short / long, all ix default to -1.
        target_byte_range = _TargetByteRange(short_start=short_answer_byte_start_ix,
                                             short_end=short_answer_byte_end_ix,
                                             long_start=long_answer.get('start_byte'),
                                             long_end=long_answer.get('end_byte'))
        print("target byte range:{}".format(target_byte_range))

    for (i, token) in enumerate(example.get('document_tokens')):
        _token = token.get('token')
        sub_tokens = tokenizer.tokenize(_token)
        start_byte = token.get('start_byte')
        end_byte = token.get('end_byte')
        for sub_token in sub_tokens:
            if not token.get('html_token'):
                all_doc_tokens.append(sub_token)
                start_bytes.append(start_byte)
                end_bytes.append(end_byte)
                token_ix.append(i)
            else:
                # TODO: Need to build the DOM for the example to infer long answers.
                pass

    start_bytes = np.array(start_bytes)
    end_bytes = np.array(end_bytes)

    # The -3 accounts for [CLS], [SEP] and [SEP]
    total_offset = len(query_tokens) + 3
    max_tokens_for_doc = max_seq_length - total_offset

    # calculate number of sliding windows based on max_seq_length and length of the doc
    def _compute_nb_spans(_all_doc_tokens):
        # Slide the window
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(_all_doc_tokens):
            length = len(_all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(_all_doc_tokens):
                break
            start_offset += length
        return doc_spans

    doc_spans = _compute_nb_spans(all_doc_tokens)
    # break the context into chunks.
    # for training, modulo the answers.
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        print(doc_span)
        tokens, segment_ids = [], []
        targets = []
        tokens.append(CLS)
        segment_ids.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(SEP)
        segment_ids.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append(SEP)
        segment_ids.append(1)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if is_training:
            # bytes for the current span
            span_start_byte, span_end_byte = start_bytes[doc_span.start], end_bytes[
                doc_span.start + doc_span.length - 1]
            print("span start byte: {}, span end byte: {}".format(span_start_byte, span_end_byte))
            # collect a single label. if short answer exists, collect. if not, check long answer.
            if target_byte_range.short_start >= span_start_byte and target_byte_range.short_end <= span_end_byte:
                print("short span")
                answer_id = 0
                # byte ix position
                s_ix = np.where(start_bytes == target_byte_range.short_start)[0]
                s = s_ix.min() - doc_span.start + total_offset - 1
                e_ix = np.where(end_bytes == target_byte_range.short_end)[0]
                e = e_ix.max() - doc_span.start + total_offset - 1
                assert 0 <= s < max_seq_length
                assert 0 <= e < max_seq_length
            elif target_byte_range.long_start >= span_start_byte and target_byte_range.long_end <= span_end_byte:
                print("long span")
                answer_id = 1
                # index. this takes into account cases where long_start coincide with HTML tags.
                s_ix = np.where(start_bytes >= target_byte_range.long_start)[0].min()
                s = s_ix - doc_span.start + total_offset - 1
                e_ix = np.where(end_bytes <= target_byte_range.long_end)[0].max()
                e = e_ix - doc_span.start + total_offset - 1
                assert 0 <= s < max_seq_length
                assert 0 <= e < max_seq_length
            else:
                answer_id = 2
                # downsample null instances if specified.
                s, e = CLS, CLS
                if downsample_null_instances:
                    if np.random.random(1) > 1. / DOWNSAMPLE:
                        continue
            print("s: {}".format(s))
            print("e: {}".format(e))
            # targets are inclusive.
            targets = (s, e)
            feature = InputFeatures(input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    targets=targets,
                                    answer_id=answer_id,
                                    tokens=tokens)
            outputs.append(feature)
        train_writer(outputs)
    return outputs


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.gfile.MakeDirs(FLAGS.output_dir)
    tf.gfile.MakeDirs(FLAGS.output_dir)

    tokenizer = create_tokenizer_from_hub_module()

    def _create_tf_records(_file_path):
        import json_lines
        examples = []
        with open(_file_path, 'rb') as f:
            for item in json_lines.reader(f):
                examples.append(item)
        file_name = 'train' if FLAGS.is_training else 'eval'
        hash_object = hashlib.md5(_file_path)
        file_name += '_' + hash_object.hexdigest()
        train_writer = FeatureWriter(
            filename=os.path.join(FLAGS.output_dir, "{}.tf_record".format(file_name)),
            is_training=FLAGS.is_training)
        for example in examples:
            convert_example(example,
                            tokenizer=tokenizer,
                            is_training=FLAGS.is_training,
                            max_seq_length=FLAGS.max_seq_length,
                            max_query_length=FLAGS.max_seq_length,
                            train_writer=train_writer)
        train_writer.close()
        del examples

    pool = multiprocessing.Pool()

    # TODO: What I am passing here?
    pool.map(_create_tf_records, range(0, 10))
    pool.close()





if __name__ == "__main__":
    tf.app.run()


