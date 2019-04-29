
import tensorflow as tf
import numpy as np

def decorate_answer(prediction, long_span, override_short_span=False):
  _prediction = {}
  _prediction['short_answers_score'] = prediction['score']
  if long_span:
    _prediction['long_answer_score'] = prediction['score']
    long_answer = {'start_byte': long_span[0], 'end_byte': long_span[1], 'start_token': -1, 'end_token': -1}
  else:
    # no answer
    _prediction['long_answer_score'] = prediction['score']
    long_answer = {'start_byte': -1, 'end_byte': -1, 'start_token': -1, 'end_token': -1}
  short_answer = {'start_byte': prediction['start_byte'], 'end_byte': prediction['end_byte'], 'start_token': -1, 'end_token': -1}
  if override_short_span:
    _prediction['short_answer'] = long_answer
  else:
    _prediction['short_answer'] = short_answer
  _prediction['long_answer'] = long_answer
  _prediction['yes_no_answer'] = 'NONE'
  _prediction['example_id'] = prediction['example_id']
  return _prediction

import pandas as pd
from collections import defaultdict

def extract_prediction(all_span_predictions, candidates, remove_scores=True):
    # get example_ids with at least one answer
    # get the highest score for example_ids with an answer
    # TODO: Should I take the prediction with most count instead?
    example_ids = set(p['example_id'] for p in all_span_predictions)
    example_ids_with_answer = set()
    example_id_answer = defaultdict(dict)
    example_id_mean_score_no_answer = defaultdict(list)
    for pred in all_span_predictions:
      if pred['start_byte'] > 0 and pred['end_byte'] > 0:
        # decorate w the long answer
        long_span = None
        for c in candidates.get(pred['example_id'], []):
          if c['top_level'] and c['start_byte'] <= pred['start_byte'] and c['end_byte'] >= pred['end_byte']:
            long_span = (c['start_byte'], c['end_byte'])
            break
        if long_span is not None:
          example_ids_with_answer.add(pred['example_id'])
          _dict = example_id_answer.get(pred['example_id'])
          answer = decorate_answer(pred, long_span)
          if _dict and _dict.get('short_answers_score') < pred['score']:
            # only update if the prediction score is higher.
            example_id_answer[pred['example_id']] = answer
          else:
            # new entry
            example_id_answer[pred['example_id']] = answer
    tf.logging.info("decorated {} examples".format(len(example_ids_with_answer)))
    # get example_ids with no answer
    # get the mean score for example_ids without an answer
    example_ids_with_no_answer = set()
    for pred in all_span_predictions:
      if pred['example_id'] not in example_ids_with_answer:
        # keep a running mean
        example_id_mean_score_no_answer[pred['example_id']].append(pred['score'])
        pred['short_answer_scores'] = np.mean(example_id_mean_score_no_answer.get(pred['example_id']))
        answer = decorate_answer(pred, None)
        # write the answer
        example_id_answer[pred['example_id']] = answer
        example_ids_with_no_answer.add(pred['example_id'])
    tf.logging.info("decorated {} examples".format(len(example_ids_with_no_answer)))
    predictions = list(example_id_answer.values())
    final_predictions = []
    if remove_scores:
      for prediction in predictions:
        prediction['short_answers_score'] = -1
        prediction['long_answer_score'] = -1
    else:
      final_predictions = predictions

    assert len(example_id_answer.keys()), len(example_ids)
    return {'predictions': predictions }