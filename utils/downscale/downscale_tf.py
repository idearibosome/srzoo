import argparse
import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
  tf.flags.DEFINE_string('input_path', 'LR', 'Base path of the input images.')
  tf.flags.DEFINE_string('output_path', 'HR', 'Base path of the output (downscaled) images.')
  tf.flags.DEFINE_integer('scale', 4, 'Downscaling factor.')


def main(unused_argv):
  # initialize
  tf.logging.set_verbosity(tf.logging.INFO)

  # downscaling session
  tf_downscale_graph = tf.Graph()
  with tf_downscale_graph.as_default():
    tf_input_path = tf.placeholder(tf.string, [])
    tf_output_path = tf.placeholder(tf.string, [])
    tf_scale = tf.placeholder(tf.int32, [])
        
    tf_image = tf.read_file(tf_input_path)
    tf_image = tf.image.decode_png(tf_image, channels=3, dtype=tf.uint8)
    tf_image = tf.image.resize_bicubic([tf_image], size=[tf.shape(tf_image)[0] // tf_scale, tf.shape(tf_image)[1] // tf_scale], align_corners=True)[0]
    tf_image = tf.cast(tf.clip_by_value(tf_image, 0.0, 255.0), tf.uint8)
    tf_image = tf.image.encode_png(tf_image)
    tf_downscale_op = tf.write_file(tf_output_path, tf_image)

    tf_downscale_init = tf.global_variables_initializer()
    tf_downscale_session = tf.Session(config=tf.ConfigProto(
      device_count={'GPU': 0}
    ))
    tf_downscale_session.run(tf_downscale_init)

  # retrieve image name list
  image_name_list = [f for f in os.listdir(FLAGS.input_path) if f.lower().endswith('.png')]
  tf.logging.info('data: %d images are prepared' % (len(image_name_list)))

  # downscale
  for (i, image_name) in enumerate(image_name_list):
    input_path = os.path.join(FLAGS.input_path, image_name)
    output_path = os.path.join(FLAGS.output_path, image_name)

    feed_dict = {
      tf_input_path: input_path,
      tf_output_path: output_path,
      tf_scale: FLAGS.scale
    }

    tf.logging.info('%d/%d, %s' % ((i+1), len(image_name_list), image_name))
    tf_downscale_session.run(tf_downscale_op, feed_dict=feed_dict)

  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()