import argparse
import os

import numpy as np
import tensorflow as tf
import cv2 as cv

FLAGS = tf.flags.FLAGS

if __name__ == '__main__':
  tf.flags.DEFINE_string('input_path', 'LR', 'Base path of the input images.')
  tf.flags.DEFINE_string('output_path', 'HR', 'Base path of the output (downscaled) images.')
  tf.flags.DEFINE_integer('scale', 4, 'Downscaling factor.')


def main(unused_argv):
  # initialize
  tf.logging.set_verbosity(tf.logging.INFO)

  # retrieve image name list
  image_name_list = [f for f in os.listdir(FLAGS.input_path) if f.lower().endswith('.png')]
  tf.logging.info('data: %d images are prepared' % (len(image_name_list)))

  # downscale
  for (i, image_name) in enumerate(image_name_list):
    input_path = os.path.join(FLAGS.input_path, image_name)
    output_path = os.path.join(FLAGS.output_path, image_name)

    tf.logging.info('%d/%d, %s' % ((i+1), len(image_name_list), image_name))

    image = cv.imread(input_path)
    image_height, image_width = image.shape[0:2]
    image = cv.resize(image, (image_width // FLAGS.scale, image_height // FLAGS.scale), interpolation=cv.INTER_CUBIC)
    cv.imwrite(output_path, image)

  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  tf.app.run()