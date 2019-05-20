import numpy as np

from .base import BaseEvaluator


def create_evaluator():
  return PSNREvaluator()


class PSNREvaluator(BaseEvaluator):

  def __init__(self):
    super().__init__()
  

  def evaluate(self, output_image, truth_image):
    # uint8 -> float64
    output_image = output_image.astype(np.float64)
    truth_image = truth_image.astype(np.float64)

    mse = np.mean((output_image - truth_image) ** 2)
    if (mse <= 0):
      return np.inf
    
    return 10 * np.log10(255.0 * 255.0 / mse)

