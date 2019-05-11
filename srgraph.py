import numpy as np
import tensorflow as tf


class SRGraph:

  DEFAULT_CONFIG = {
    'channel_first': False,
    'input_name': 'sr_input',
    'input_scale_name': 'sr_input_scale',
    'output_name': 'sr_output',
    'pixel_range': [0.0, 255.0],
    'use_scale_placeholder': False
  }


  def __init__(self):
    pass
  

  def prepare(self, scale, standalone, config, model_path):
    self.scale = scale
    self.standalone = standalone
    self.model_path = model_path

    # config
    self.config = config
    for (key, value) in self.DEFAULT_CONFIG.items():
      if (not key in self.config):
        self.config[key] = value
    
    # build a standalone graph
    if (self.standalone):
      self.tf_graph = tf.Graph()
      with self.tf_graph.as_default():
        self.tf_input = tf.placeholder(tf.float32, [None, None, None, 3])
        self.tf_output = self._get_output(self.tf_input)

        self.tf_init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

      self.tf_session = tf.Session(
        config=tf.ConfigProto(
          log_device_placement=False,
          allow_soft_placement=True
        ),
        graph=self.tf_graph
      )
      self.tf_session.run(self.tf_init_op)
  

  def get_output(self, input_list):
    if (self.standalone):
      feed_dict = {}
      feed_dict[self.tf_input] = input_list

      output_list = self.tf_session.run(self.tf_output, feed_dict=feed_dict)
      return output_list
    
    return self._get_output(input_list)


  def _get_output(self, input_list):

    # load model graph
    with tf.gfile.GFile(self.model_path, 'rb') as f:
      model_graph_def = tf.GraphDef()
      model_graph_def.ParseFromString(f.read())
    
    return self._sr_graph(graph_def=model_graph_def, input_list=input_list)
    

  def _sr_graph(self, graph_def, input_list):
    x = input_list

    # adjust channel dimension
    if (self.config['channel_first']):
      x = tf.transpose(x, [0, 3, 1, 2])
    
    # adjust pixel range
    # assume that the input has a range of [0, 255] (uint8)
    x = (x * ((self.config['pixel_range'][1] - self.config['pixel_range'][0]) / 255.0)) + self.config['pixel_range'][0]

    # build input map
    input_map = {}
    input_map[self.config['input_name']+':0'] = x
    if (self.config['use_scale_placeholder']):
      input_map[self.config['input_scale_name']+':0'] = self.scale
    
    # obtain output from graph_def
    y = tf.import_graph_def(graph_def, name='model', input_map=input_map, return_elements=[self.config['output_name']+':0'])[0]

    # adjust pixel range to [0, 255]
    y = (y - self.config['pixel_range'][0]) * (255.0 / (self.config['pixel_range'][1] - self.config['pixel_range'][0]))

    # adjust channel dimension
    if (self.config['channel_first']):
      y = tf.transpose(y, [0, 2, 3, 1])
    
    return y
        

