

def create_evaluator():
  return BaseEvaluator()


class BaseEvaluator:


  def __init__(self):
    pass
  

  def evaluate(self, output_image, truth_image):
    """
    Evaluate the performance from the given output image and truth image.
    Args:
      output_image: The output image obtained from the super-resolution model.
      truth_image: The ground-truth image that is expected to be similar to output_image.
    Returns:
      value: The evaluated output value.
    """
    raise NotImplementedError

