import torch

import numpy as np
import tensorflow as tf

from pytorch2keras.converter import pytorch_to_keras

import converter_common

def convert_to_srzoo(model, input_name='sr_input', channels_first=True, model_name='model.pb', output_name='sr_output'):

  if (channels_first):
    input_np = np.zeros([1, 3, 32, 32])
    input_shape = (3, None, None)
  else:
    input_np = np.zeros([1, 32, 32, 3])
    input_shape = (None, None, 3)
  
  input_var = torch.autograd.Variable(torch.FloatTensor(input_np))

  keras_model = pytorch_to_keras(model, (input_var), input_names=[input_name], input_shapes=[input_shape], verbose=True)

  sess = tf.keras.backend.get_session()
  output = keras_model.outputs[0]

  with sess.graph.as_default():
    output_node = tf.identity(output, name=output_name)
  
  converter_common.write_tf_session_graph(sess=sess, model_name=model_name, output_name=output_name)
