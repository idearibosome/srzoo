import argparse
import csv
import importlib
import json
import os
import time

import numpy as np
import tensorflow as tf

from utils import image_utils


# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--sr_path', default='SR', help='folder path of the high resolution (output) images')
parser.add_argument('--truth_path', default='HR', help='folder path of the high resolution (ground-truth) images')
parser.add_argument('--shave_borders', type=int, default=4, help='amount of pixels to shave borders')
parser.add_argument('--color_mode', default='ycbcry', help='color mode (ycbcry: Y channel of YCbCr, rgb: RGB channels of RGB)')
parser.add_argument('--evaluators', default='psnr,ssim', help='list of evaluators (seperate with comma(,))')
parser.add_argument('--output_name', default='evaluate.csv', help='file name of the evaluation results (csv format)')
args = parser.parse_args()


# constants
IMAGE_EXTS = ['.png', '.jpg']


def main():
  # initialize
  tf.logging.set_verbosity(tf.logging.INFO)

  # image utils
  image_reader = image_utils.ImageReader()
  image_manipulator = image_utils.ImageManipulator()

  # evaluators
  evaluator_name_list = args.evaluators.split(',')
  evaluator_dict = {}
  for evaluator_name in evaluator_name_list:
    evaluator_module = importlib.import_module('evaluators.%s' % (evaluator_name))
    evaluator = evaluator_module.create_evaluator()
    evaluator_dict[evaluator_name] = evaluator

  # image path list
  image_path_list = []
  for root, _, files in os.walk(args.sr_path):
    for filename in files:
      for ext in IMAGE_EXTS:
        if (filename.lower().endswith(ext)):
          image_name = os.path.splitext(filename)[0]
          sr_path = os.path.join(root, filename)
          truth_path = os.path.join(args.truth_path, filename)

          image_path_list.append([image_name, sr_path, truth_path])
  tf.logging.info('found %d images' % (len(image_path_list)))

  # iterate
  evaluation_list = []
  for image_name, sr_path, truth_path in image_path_list:
    output_image = image_reader.read(sr_path)
    truth_image = image_reader.read(truth_path)

    # to floating point
    output_image = output_image.astype(np.float64)
    truth_image = truth_image.astype(np.float64)

    # color channels
    if (args.color_mode == 'ycbcry'):
      output_image = image_manipulator.rgb_to_ycbcr(output_image)[:, :, 0:1]
      truth_image = image_manipulator.rgb_to_ycbcr(truth_image)[:, :, 0:1]

    # crop
    output_image = image_manipulator.shave_border(output_image, args.shave_borders)
    truth_image = image_manipulator.shave_border(truth_image, args.shave_borders)
    output_image, truth_image = image_manipulator.match_size(output_image, truth_image)

    evaluation = {}
    evaluation['name'] = image_name

    tf.logging.info('%s', image_name)

    for evaluator_name in evaluator_name_list:
      evaluator = evaluator_dict[evaluator_name]
      evaluation[evaluator_name] = evaluator.evaluate(output_image=output_image, truth_image=truth_image)
      tf.logging.info('- %s: %s' % (evaluator_name, str(evaluation[evaluator_name])))

    evaluation_list.append(evaluation)
  
  # save result
  if (args.output_name is not None):
    with open(args.output_name, 'w', newline='') as csvfile:
      fieldnames = ['name']
      fieldnames.extend(evaluator_name_list)

      writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

      writer.writeheader()
      for evaluation in evaluation_list:
        writer.writerow(evaluation)
  
  # finalize
  tf.logging.info('finished')
  
  # average values
  for evaluator_name in evaluator_name_list:
    evaluator_evaluation_list = [x[evaluator_name] for x in evaluation_list]
    tf.logging.info('- %s (average): %s' % (evaluator_name, str(np.mean(evaluator_evaluation_list))))


if __name__ == '__main__':
  main()
