import numpy as np
import tensorflow as tf

from .base import BaseEvaluator


def create_evaluator():
  return SSIMEvaluator()


class SSIMEvaluator(BaseEvaluator):

  def __init__(self):
    super().__init__()

    # graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      self.tf_image1 = tf.placeholder(tf.float64, [None, None, None])
      self.tf_image2 = tf.placeholder(tf.float64, [None, None, None])

      self.tf_ssim = tf.image.ssim(self.tf_image1, self.tf_image2, max_val=255.0)

      init = tf.global_variables_initializer()
      self.tf_session = tf.Session(config=tf.ConfigProto(
          device_count={'GPU': 0}
      ))
      self.tf_session.run(init)
  

  def evaluate(self, output_image, truth_image):
    # uint8 -> float64
    output_image = output_image.astype(np.float64)
    truth_image = truth_image.astype(np.float64)

    ssim = self.tf_session.run(self.tf_ssim, feed_dict={self.tf_image1:output_image, self.tf_image2:truth_image})
    
    return ssim

