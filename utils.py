""" Utility functions. """
import numpy as np
import os
import random
from scipy.misc import imread, imresize
from scipy.ndimage import rotate, shift
from skimage import transform
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

import matplotlib
matplotlib.use('Agg')
from matplotlib.pyplot import imsave

FLAGS = flags.FLAGS

## Image helper
def get_images(paths, labels, nb_samples=None, shuffle=True):
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x
    images = [(i, os.path.join(path, image)) \
        for i, path in zip(labels, paths) \
        for image in sampler(os.listdir(path))]
    if shuffle:
        random.shuffle(images)
    return images

## Network helpers
def conv_block(inp, cweight, bweight, reuse, scope, bn_vars=None, incl_stride=True, activation=tf.nn.relu):
    """ Perform, conv, batch norm, nonlinearity, and max pool """
    stride, no_stride = [1,2,2,1], [1,1,1,1]

    if FLAGS.max_pool or not incl_stride:
        conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
    else:
        conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
    normed = normalize(conv_output, activation, reuse, scope, bn_vars=bn_vars)
    if FLAGS.max_pool and incl_stride:
        normed = tf.nn.max_pool(normed, stride, stride, 'VALID')
    return normed

def normalize(inp, activation, reuse, scope, bn_vars=None):
    if FLAGS.norm == 'batch_norm':
        if bn_vars == None:
            return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        else:
            scale, offset = bn_vars
            normed = tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope, scale=False, center=False)
            return offset+normed*scale # TODO
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        return activation(inp)

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def l1_loss(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.abs(pred-label))


def xent(pred, label):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

def sigmoid_xent(pred, label):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def load_transform(image_path, angle=0., s=(0, 0), size=(20, 20), scale=None):
  # Load the image
  original = imread(image_path) #, flatten=True)
  original = 255 - original

  scale = 1.0, 1.0 #1.1111111, 1.1111111 #0.9, 0.9 #1.25, 1.25 #0.8, 0.8 # corresponds to 1.25x larger
  shear = 0 #-np.pi/6
  if scale:
    orig_size = original.shape[0]
    T = np.array([[1, 0,-orig_size/2.0],[0, 1,-orig_size/2.0],[0, 0, 1]])
    invT = np.linalg.inv(T)
    tform = transform.AffineTransform(scale=scale, shear=shear)
    new_params = invT.dot(tform.params).dot(T)
    tform = transform.AffineTransform(new_params)
    original = transform.warp(original, tform)

  # Rotate the image
  rotated = rotate(original, angle=angle, cval=0.)
  # Shift the image
  #shifted = shift(rotated, shift=s)
  # Resize the image
  resized = imresize(rotated, size=size, interp='lanczos')
  #resized = imresize(rotated, size=size)
  resized = np.asarray(resized, dtype=np.float32) / 255
  #return resized
  # Invert the image
  #inverted = 1. - resized
  return resized


