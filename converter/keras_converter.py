import tensorflow as tf

import converter_common

def convert_to_srzoo(model, model_name='model.pb', output_name='sr_output'):

  sess = tf.keras.backend.get_session()
  output = model.outputs[0]

  with sess.graph.as_default():
    output_node = tf.identity(output, name=output_name)
  
  converter_common.write_tf_session_graph(sess=sess, model_name=model_name, output_name=output_name)
