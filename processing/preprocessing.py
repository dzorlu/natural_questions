
import tensorflow as tf
import tensorflow_hub as hub
import collections
import numpy as np
import os
from bert import tokenization

import re

# This is a path to an uncased (all lowercase) version of BERT
BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
CLS = "[CLS]"
SEP = "[SEP]"

DOWNSAMPLE = 100
BETA = 2 #for evalution
# context_id special tokens must be in the vocabulary, and hence limited to a number.
# if it exceeds the max id, move on to the next example.
MAX_CONTEXT_ID = 50


flags = tf.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_data_dir", None,
    "The output directory where the tf records will be written.")

flags.DEFINE_string(
    "nq_vocab_file", None,
    "The path to vocab file that includes special tokens.")


flags.DEFINE_integer(
    "max_seq_length", None,
    "max length")

flags.DEFINE_integer(
    "doc_stride", None,
    "doc_stride")


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, mode):
    self.filename = filename
    self.mode = mode
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """
    Write a InputFeature to the TFRecordWriter as a tf.train.Example.
    """
    self.num_features += 1
    if self.num_features % 1e3 == 0:
        tf.logging.info((self.num_features, self.filename))

    def create_int_feature(values):
      feature = tf.train.Feature(
          int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["example_id"] = create_int_feature([feature.example_id])
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["start_bytes"] = create_int_feature(feature.start_bytes)
    features["end_bytes"] = create_int_feature(feature.end_bytes)


    if self.mode in ['eval','train']:
      positions = list(sum(feature.targets, ()))
      features["positions"] = create_int_feature(positions)
      features['answer_id'] = create_int_feature(feature.answer_id)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())
    return

  def close(self):
    tf.logging.info("{}: {} examples found".format(self.filename, str(self.num_features)))
    self._writer.close()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 example_id,
                 input_ids,
                 input_mask,
                 segment_ids,
                 tokens,
                 start_bytes,
                 end_bytes,
                 answer_id=None,
                 targets=None,
                 ):
        self.example_id = example_id
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.targets = targets
        self.tokens = tokens
        self.answer_id = answer_id
        self.start_bytes = start_bytes
        self.end_bytes = end_bytes

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        if self.example_id:
            s += "example_id: {}".format(self.example_id)
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
        if self.start_bytes:
            s += ", start_bytes: {}".format(self.start_bytes)
        if self.end_bytes:
            s += ", end_bytes: {}".format(self.end_bytes)
        return s


def create_tokenizer(vocab_file):
    """
    Get the modified vocab file. 
    Includes special tokens [context_id=n] where n <= 50
    """
    return tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)


def convert_example(example,
                    tokenizer,
                    max_seq_length,
                    doc_stride,
                    max_query_length=None,
                    mode='train',
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

    def _get_target_byte_range(_annotation):
        # short answers
        short_answers = _annotation.get('short_answers')
        short_answer_byte_start_ix, short_answer_byte_end_ix = -1, -1
        if short_answers:
            # if all annotated short spans are contained in the instance,
            # we set the start and end target indices to point to the
            # smallest span containing all annotated short answer spans
            short_answer_byte_start_ix = min([sa.get('start_byte') for sa in short_answers])
            short_answer_byte_end_ix = max([sa.get('end_byte') for sa in short_answers])
        # long answers
        long_answer = _annotation.get('long_answer')
        # if no short / long, all ix default to -1.
        target_byte_range = _TargetByteRange(short_start=short_answer_byte_start_ix,
                                             short_end=short_answer_byte_end_ix,
                                             long_start=long_answer.get('start_byte'),
                                             long_end=long_answer.get('end_byte'))
        return target_byte_range

    if mode == 'train':
        annotation = next(iter(example.get('annotations')))
        target_byte_ranges = [_get_target_byte_range(annotation)]
    elif mode == 'eval':
        # eval data provides 5-way answers.
        target_byte_ranges = []
        for annotation in example.get('annotations'):
            target_byte_ranges.append(_get_target_byte_range(annotation))

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

    start_bytes = np.array(start_bytes)
    end_bytes = np.array(end_bytes)

    # The -3 accounts for [CLS], [SEP] and [id=n]
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
            start_offset += min(length, doc_stride)
        return doc_spans


    # context_id supervision - for each span, tell the model the index of long answer candidate
    # top level candidates does not cover the entire text. I use start_byte as the index to ensure
    # to assign a context_id for each span.
    top_level_candidates = [c for c in example['long_answer_candidates'] if c['top_level']]
    # [(context_id, start_byte), ...]
    context_ids = [(i, c['start_byte']) for i, c in enumerate(top_level_candidates)]

    doc_spans = _compute_nb_spans(all_doc_tokens)
    # break the context into chunks.
    # for training, modulo the answers.
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens, segment_ids = [], []
        start_bytes_span, end_bytes_span = [], [] # used for prediction and eval
        tokens.append(CLS)
        segment_ids.append(0)
        start_bytes_span.append(0)
        end_bytes_span.append(0)
        for token in query_tokens:
            tokens.append(token)
            segment_ids.append(0)
            start_bytes_span.append(0)
            end_bytes_span.append(0)
        tokens.append(SEP)
        segment_ids.append(0)
        start_bytes_span.append(0)
        end_bytes_span.append(0)
        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
            _start_byte = start_bytes[split_token_index]
            start_bytes_span.append(_start_byte)
            end_bytes_span.append(end_bytes[split_token_index])
        # the context_id token is appended at the end. this does not use the tokenizer but because
        # the context_id variety is already in the vocabulary, convert_tokens_to_ids below works.
        # determine the context_id given start_byte
        # this lumps the text that preceeds first start_byte into the same context_id with text
        # that comes after first start_byte.
        def _get_context_id(_span_start_byte):
            context_id = [c[0] for c in context_ids if c[1] > _span_start_byte]
            # if null, return zero. if context_id exceeds MAX_CONTEXT_ID, return the same token
            context_id = context_id[0] if context_id else 0
            if context_id > MAX_CONTEXT_ID:
                context_id = 99
            return "[context_id=%s]" % context_id
        span_start_byte = start_bytes[doc_span.start]
        #TODO: Do I need a SEP token to end the span?
        tokens.append(_get_context_id(span_start_byte))
        segment_ids.append(1)
        start_bytes_span.append(0)
        end_bytes_span.append(0)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
          input_ids.append(0)
          input_mask.append(0)
          segment_ids.append(0)
          start_bytes_span.append(0)
          end_bytes_span.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(start_bytes_span) == max_seq_length
        assert len(end_bytes_span) == max_seq_length

        if mode in ['train','eval']:
          answer_ids = []
          targets = []
          s_ix, e_ix = None, None
          # bytes for the current span
          span_start_byte, span_end_byte = start_bytes[doc_span.start], end_bytes[
              doc_span.start + doc_span.length - 1]
          for target_byte_range in target_byte_ranges:
              answer_id = None
              # collect a single label. if short answer exists, collect. if not, check long answer.
              if target_byte_range.short_start >= span_start_byte and target_byte_range.short_end <= span_end_byte:
                  answer_id = 0
                  # byte ix position
                  s_ix = np.where(start_bytes == target_byte_range.short_start)[0]
                  e_ix = np.where(end_bytes == target_byte_range.short_end)[0]
              elif target_byte_range.long_start >= span_start_byte and target_byte_range.long_end <= span_end_byte:
                  answer_id = 1
                  # index. this takes into account cases where long_start coincide with HTML tags.
                  s_ix = np.where(start_bytes >= target_byte_range.long_start)[0]
                  e_ix = np.where(end_bytes <= target_byte_range.long_end)[0]
              if answer_id in (0, 1) and s_ix is not None and e_ix is not None:
                try:
                  s = s_ix.min() - doc_span.start + total_offset - 1
                  e = e_ix.max() - doc_span.start + total_offset - 1
                except:
                  tf.logging.info('error encountered..')
                  continue
                if not 0 <= e < max_seq_length or not 0 <= s <= max_seq_length:
                  # this ensures the first/last token for the byte is included in the current span.
                  tf.logging.info("{} is outside the range".format((s,e)))
                  continue
                # targets are inclusive.
                targets.append((s, e))
                answer_ids.append(answer_id)
          # for eval, if less than 2 annotations are found, discard.
          if mode == 'eval':
            if len(targets) < BETA:
              targets = []

          if downsample_null_instances and not targets:
            # downsample null instances if specified.
            if np.random.random(1) > 1. / DOWNSAMPLE:
              continue
            _len = 5 if mode == 'eval' else 1
            answer_ids = _len * [2]
            # no answer
            targets = _len *  [(0, 0)]
          # pad eval targets
          if mode == 'eval' and len(targets) >= BETA:
            while len(targets) < 5:
              targets.append((-1, -1))
              answer_ids.append(-1)
          if targets and answer_ids:
            feature = InputFeatures(example_id=example.get('example_id'),
                                    input_ids=input_ids,
                                    input_mask=input_mask,
                                    segment_ids=segment_ids,
                                    targets=targets,
                                    answer_id=answer_ids,
                                    start_bytes=start_bytes_span,
                                    end_bytes=end_bytes_span,
                                    tokens=tokens)
            if train_writer:
              train_writer(feature)
        else:
          # predict
          feature = InputFeatures(example_id=example.get('example_id'),
                                  input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  start_bytes=start_bytes_span,
                                  end_bytes=end_bytes_span,
                                  tokens=tokens)
          if train_writer: train_writer(feature)


def main(_):
  import jsonlines
  import re
  tf.logging.set_verbosity(tf.logging.INFO)

  _dev_path = os.path.join(FLAGS.bert_data_dir, 'dev')
  _train_path = os.path.join(FLAGS.bert_data_dir, 'train')
  [tf.gfile.MakeDirs(_dir) for _dir in [_train_path, _dev_path]]

  tokenizer = create_tokenizer(FLAGS.nq_vocab_file)

  def _create_tf_records(mode, _train_file):
    tf.logging.info(_train_file)
    # writes into the same directory
    _train_file_out = re.sub(".jsonl", "_{}.tf_record".format(mode), _train_file)
    train_writer = FeatureWriter(
        filename=_train_file_out,
        mode=mode)
    print("file {}".format(_train_file))
    with jsonlines.open(_train_file) as reader:
        for i, example in enumerate(reader):
            if i % 1e3 == 0: tf.logging.info("{}:{}".format(_train_file, str(i)))
            convert_example(example,
                            tokenizer=tokenizer,
                            mode=mode,
                            max_seq_length=FLAGS.max_seq_length,
                            doc_stride=FLAGS.doc_stride,
                            max_query_length=FLAGS.max_seq_length,
                            train_writer=train_writer.process_feature)

    tf.logging.info("{} completed".format(_train_file))
    tf.logging.info("{}:{} examples processed".format(mode, train_writer.num_features))
    train_writer.close()
  train_files = [os.path.join(_train_path, _file) for _file in os.listdir(_train_path) if _file.endswith(".jsonl")]
  dev_files = [os.path.join(_dev_path, _file) for _file in os.listdir(_dev_path) if _file.endswith(".jsonl")]
  # extract targets
  for train_file in train_files:
    _create_tf_records('train', train_file)
  for dev_file in dev_files:
    _create_tf_records('eval', dev_file)
  # for inference
  for dev_file in dev_files:
    _create_tf_records('inference', dev_file)

if __name__ == "__main__":
    tf.app.run()


