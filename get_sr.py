import argparse
import json
import os

import numpy as np
import tensorflow as tf

from utils import image_utils
from srgraph import SRGraph


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config_path', required=True, help='path of the config file (.json)')
parser.add_argument('--model_path', required=True, help='path of the model file (.pb)')
parser.add_argument('--input_path', default='LR', help='folder path of the lower resolution (input) images')
parser.add_argument('--output_path', default='SR', help='folder path of the high resolution (output) images')
parser.add_argument('--scale', default=4, help='upscaling factor')
parser.add_argument('--cuda_device', default='-1', help='CUDA device index to be used (will be set to the environment variable \'CUDA_VISIBLE_DEVICES\')')
args = parser.parse_args()


# constants
IMAGE_EXTS = ['.png', '.jpg']


def main():
  # initialize
  os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_device
  tf.logging.set_verbosity(tf.logging.INFO)

  # SR config
  with open(args.config_path, 'r') as f:
    sr_config = json.load(f)

  # SR graph
  sr_model = SRGraph()
  sr_model.prepare(scale=args.scale, standalone=True, config=sr_config, model_path=args.model_path)

  # image reader/writer
  image_reader = image_utils.ImageReader()
  image_writer = image_utils.ImageWriter()

  # image path list
  image_path_list = []
  for root, _, files in os.walk(args.input_path):
    for filename in files:
      for ext in IMAGE_EXTS:
        if (filename.lower().endswith(ext)):
          image_name = os.path.splitext(filename)[0]
          input_path = os.path.join(root, filename)
          output_path = os.path.join(args.output_path, '%s.png' % (image_name))

          image_path_list.append([input_path, output_path])
  tf.logging.info('found %d images' % (len(image_path_list)))

  # iterate
  for input_path, output_path in image_path_list:
    tf.logging.info('%s -> %s' % (input_path, output_path))
    input_image = image_reader.read(input_path)
    output_image = sr_model.get_output([input_image])[0]
    output_image = np.clip(output_image, 0, 255)
    image_writer.write(output_image, output_path)
  
  # finalize
  tf.logging.info('finished')


if __name__ == '__main__':
  main()
