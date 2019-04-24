
import pandas as pd

def extract_prediction(predictions, candidates):
  # take the max value of span answers. remove non-answers
  # TODO: max count instead of max value.
  predictions = pd.DataFrame(predictions)
  predictions = predictions[predictions['start_byte'] != 0]
  predictions_max = predictions.sort_values('score', ascending=False).groupby('example_id').head(1)
  final_predictions = []
  for _, p in predictions_max.iterrows():
    long_span = None
    for c in candidates.get(p['example_id'], []):
      if c['top_level'] and c['start_byte'] <= p['start_byte'] and c['end_byte'] >= p['end_byte']:
        long_span = (c['start_byte'], c['end_byte'])
        break
    if long_span is not None:
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
  return {'predictions': final_predictions}