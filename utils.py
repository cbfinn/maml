""" Utility functions. """
import numpy as np
import os
import random
from scipy.misc import imread, imresize
from scipy.ndimage import rotate, shift
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

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
# This is called on a per task basis
def mse(pred, label, sine_x=None, loss_weights=None, postupdate=None):
    if FLAGS.pred_task and postupdate == False:
        if FLAGS.learn_loss:
            #inp = tf.concat([pred, label[:, 1:]], 1)
            if FLAGS.label_in_loss:
                inp = tf.concat([sine_x, pred, label[:, :]], 1)
            else:
                inp = tf.concat([sine_x, pred, label[:, 1:]], 1)
            hidden = tf.nn.relu(tf.matmul(inp, loss_weights['lw1']) + loss_weights['lb1'])
            hidden = tf.nn.relu(tf.matmul(hidden, loss_weights['lw2']) + loss_weights['lb2'])
            preloss = tf.square(tf.matmul(hidden, loss_weights['lw3']) + loss_weights['lb3'])
            learned_loss = tf.reduce_mean(preloss)
            if FLAGS.semisup_loss: 
                #true_loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(label, [-1])))
                # fix to only predict y, not task
                true_loss = tf.reduce_mean(tf.square(tf.reshape(pred[:,0], [-1]) - tf.reshape(label[:,0], [-1])))
                if FLAGS.train:
                    pick_loss = tf.cast(tf.random_uniform((1,)) > 0.5, tf.float32)
                else: 
                    pick_loss = 1.0
                return pick_loss*learned_loss + (1.0-pick_loss)*true_loss
            else:
                return learned_loss
        else:
            # only use last two entries (amp and phase)
            pred = pred[:, 1:]
            label = label[:, 1:]
    elif FLAGS.pred_task and postupdate == True:
        pred = pred[:, 0]
        label = label[:, 0]
    elif FLAGS.pred_task:
        raise ValueError('post update var must be set in mse loss')

    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))

def xent(pred, label, **kwargs):
    # Note - with tf version <=0.12, this loss has incorrect 2nd derivatives
    label *= 0.9
    label += 0.01
    return tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size

def sigmoid_xent(pred, label, **kwargs):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=label) / FLAGS.update_batch_size


def load_transform(image_path, angle=0., s=(0, 0), size=(20, 20)):
  # Load the image
  original = imread(image_path, flatten=True)
  # Rotate the image
  rotated = rotate(original, angle=angle, cval=1.)
  # Shift the image
  shifted = shift(rotated, shift=s)
  # Resize the image
  resized = np.asarray(imresize(shifted, size=size, interp='lanczos'), dtype=np.float32) / 255.
  #return resized
  # Invert the image
  inverted = 1. - resized
  return inverted

def load_transform_color(image_path, angle=0., s=(0, 0), size=(20, 20)):
    original = np.float32(imread(image_path))
    assert np.max(original) > 1.
    original /= 255.

    rotated = np.maximum(np.minimum(rotate(original, angle=angle, cval=1.), 1.), 0.)
    s = (s[0], s[1], 0)
    shifted = shift(rotated, shift=s)
    resized = imresize(shifted, size=size, interp='lanczos')

    return resized






