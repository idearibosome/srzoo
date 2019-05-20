import tensorflow as tf

def write_tf_session_graph(sess, model_name='model.pb', output_name='sr_output'):
  graph_def = sess.graph.as_graph_def()

  # remove device info in graphdef
  for node in graph_def.node:
    node.device = ''
  
  # convert variables to constants
  constant_graph = tf.graph_util.convert_variables_to_constants(sess, graph_def, output_node_names=[output_name])

  # write graph
  tf.io.write_graph(constant_graph, '.', model_name, as_text=False)
