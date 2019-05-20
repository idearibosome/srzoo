import numpy as np
import tensorflow as tf


class ImageReader:

  def __init__(self):
    # image reading graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      self.tf_image_path = tf.placeholder(tf.string, [])
      
      tf_image = tf.read_file(self.tf_image_path)
      tf_image = tf.image.decode_image(tf_image, channels=3, dtype=tf.uint8)
      
      self.tf_image = tf_image

      init = tf.global_variables_initializer()
      self.tf_session = tf.Session(config=tf.ConfigProto(
          device_count={'GPU': 0}
      ))
      self.tf_session.run(init)
  
  def read(self, image_path):
    image = self.tf_session.run(self.tf_image, feed_dict={self.tf_image_path:image_path})
    return image


class ImageWriter:

  def __init__(self):
    # image writing graph
    self.tf_graph = tf.Graph()
    with self.tf_graph.as_default():
      self.tf_image = tf.placeholder(tf.uint8, [None, None, 3])
      self.tf_image_path = tf.placeholder(tf.string, [])

      tf_image = tf.image.encode_png(self.tf_image)
      tf_write_op = tf.write_file(self.tf_image_path, tf_image)

      self.tf_write_op = tf_write_op

      init = tf.global_variables_initializer()
      self.tf_session = tf.Session(config=tf.ConfigProto(
          device_count={'GPU': 0}
      ))
      self.tf_session.run(init)
  
  def write(self, image, image_path):
    self.tf_session.run(self.tf_write_op, feed_dict={self.tf_image:image, self.tf_image_path:image_path})


class ImageManipulator:

  def __init__(self):
    pass
  
  def match_size(self, image1, image2):
    image1_shape = image1.shape
    image2_shape = image2.shape

    image1 = image1[:min(image1_shape[0], image2_shape[0]), :min(image1_shape[1], image2_shape[1])]
    image2 = image2[:min(image1_shape[0], image2_shape[0]), :min(image1_shape[1], image2_shape[1])]

    return image1, image2
  
  def shave_border(self, image, amount):
    image = image[amount:-amount, amount:-amount]
    return image
  
  def rgb_to_ycbcr(self, rgb):
    ycbcr = np.zeros_like(rgb)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    ycbcr[:, :, 0] = 16.0 + (65.481/255.0 * r) + (128.553/255.0 * g) + (24.966/255.0 * b)
    ycbcr[:, :, 1] = 128.0 - (37.797/255.0 * r) - (74.203/255.0 * g) + (112.0/255.0 * b)
    ycbcr[:, :, 2] = 128.0 - (112.0/255.0 * r) - (93.786/255.0 * g) - (18.214/255.0 * b)

    return ycbcr
  