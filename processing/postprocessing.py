import pandas as pd
import tensorflow as tf


def decorate_null_answer(example_id):
    return {
        'example_id': int(example_id),
        'long_answer': {
            'start_byte': -1,
            'end_byte': -1,
            'start_token': -1,
            'end_token': -1
        },
        'long_answer_score': -1,  # should this be a score?
        'short_answer': {
            'start_byte': -1,
            'end_byte': -1,
            'start_token': -1,
            'end_token': -1
        },
        'short_answers_score': -1,
        'yes_no_answer': 'NONE'
    }


def extract_prediction(predictions, candidates):
    # take the max value of span answers. remove non-answers
    # TODO: max count instead of max value.
    predictions = pd.DataFrame(predictions)
    example_ids = set(predictions['example_id'])
    processed_example_ids = set()
    predictions = predictions[(predictions['end_byte'] > 0) & (predictions['start_byte'] > 0)]
    # annotate examples with no span predictions as no answer.
    processed_example_ids = example_ids - set(predictions['example_id'])
    final_predictions = [decorate_null_answer(_example_id) for _example_id in processed_example_ids]
    # take the max. TODO: do count first.
    # take top 5 scores because some predictions extend beyond candidate spans.
    predictions_max = predictions.sort_values('score', ascending=False).groupby('example_id').head(5)
    for _, p in predictions_max.iterrows():
        if p['example_id'] not in processed_example_ids:
          long_span = None
          for c in candidates.get(p['example_id'], []):
            if c['top_level'] and c['start_byte'] <= p['start_byte'] and c['end_byte'] >= p['end_byte']:
              long_span = (c['start_byte'], c['end_byte'])
              break
          if long_span is not None:
            processed_example_ids.add(int(p['example_id']))
            prediction = {
              'example_id': int(p['example_id']),
              'long_answer': {
                'start_byte': int(long_span[0]),
                'end_byte': int(long_span[1]),
                'start_token': -1,
                'end_token': -1
                },
                'long_answer_score': float(p['score']),
                'short_answer': {
                  'start_byte': int(p['start_byte']),
                  'end_byte': int(p['end_byte']),
                  'start_token': -1,
                  'end_token': -1
                },
                'short_answers_score': float(p['score']),
                'yes_no_answer': 'NONE'
            }
            final_predictions.append(prediction)
    # append missing ones. need to have a better solution
    missing_example_ids = example_ids - processed_example_ids
    tf.logging.info("decorating {} missing examples".format(len(missing_example_ids)))
    final_predictions += [decorate_null_answer(_example_id) for _example_id in missing_example_ids]
    return {'predictions': final_predictions}