import argparse
import json
import os
import time

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
parser.add_argument('--self_ensemble', action='store_true', help='employ self ensemble')
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

  # run a dummy image to initialize internal graph
  input_image = np.zeros([32, 32, 3], dtype=np.uint8)
  sr_model.get_output([input_image])

  # iterate
  running_time_list = []
  for input_path, output_path in image_path_list:
    input_image = image_reader.read(input_path)

    running_time = 0.0

    if (args.self_ensemble):
      output_images = []
      ensemble_running_time_list = []
      for flip_index in range(2): # for flipping
        input_image = np.transpose(input_image, axes=(1, 0, 2))

        for rotate_index in range(4): # for rotating
          input_image = np.rot90(input_image, k=1, axes=(0, 1))

          t1 = time.perf_counter()
          output_image = sr_model.get_output([input_image])[0]
          t2 = time.perf_counter()
          ensemble_running_time_list.append(t2 - t1)

          output_image = np.clip(output_image, 0, 255)

          output_image = np.rot90(output_image, k=(3-rotate_index), axes=(0, 1))
          if (flip_index == 0):
            output_image = np.transpose(output_image, axes=(1, 0, 2))
          output_images.append(output_image)
      
      output_image = np.mean(output_images, axis=0)
      running_time = np.sum(ensemble_running_time_list)
    
    else:
      t1 = time.perf_counter()
      output_image = sr_model.get_output([input_image])[0]
      t2 = time.perf_counter()
      running_time = (t2 - t1)

      output_image = np.clip(output_image, 0, 255)
    
    output_image = np.round(output_image)
    
    image_writer.write(output_image, output_path)
    tf.logging.info('%s -> %s, %.3f sec' % (input_path, output_path, running_time))
    running_time_list.append(running_time)
  
  # finalize
  tf.logging.info('finished')
  tf.logging.info('averaged running time per image: %.3f sec' % (np.mean(running_time_list)))


if __name__ == '__main__':
  main()
