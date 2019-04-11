import tensorflow as tf
from run_nq import argmax_2d, span_accuracy
import numpy as np

tf.enable_eager_execution()

start_l = tf.constant([[0.1, 0.4, 0.5],[0.1, 0.4, 0.5],[0.1, 0.9, 0.5]])
end_l = tf.constant([[0.1, 0.4, 0.5],[0.5, 0.4, 0.1],[0.1, 0.9, 0.5]])

y_pred, y_pred_ix = argmax_2d(start_l, end_l)
assert y_pred, np.array([2.7182817, 2.2255406, 6.0496473])
assert y_pred_ix, np.array([[2, 2], [1, 1], [1, 1]])


n_way = 5
predictions = tf.constant([[1,2],[10,12],[30,30]])
positions = tf.constant([[[1,2],[4,5],[3,5],[3,5],[1,5]],
                        [[10,11],[10,11],[9,12],[3,5],[3,5]],
                        [[30,31],[29,39],[9,12],[3,5],[3,5]]])


_accuracy = span_accuracy(predictions, positions)
assert _accuracy, np.array([True, False, False])