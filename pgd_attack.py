"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import argparse # adding parser for config file
parser = argparse.ArgumentParser(description='Train an adversarial network according to the specified config file')
parser.add_argument('-c', '--config', type=str, default='config.json', help='path to the config file to train the adversarial network')
args = parser.parse_args()


class PGDAttack:
  def __init__(self, model, norm, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.norm = norm
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      self.loss = model.y_xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              10,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      self.loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      self.loss = model.y_xent

    self.grad = tf.gradients(self.loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess, iters=1):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""

    best_x = x_nat
    best_loss = sess.run(self.loss, feed_dict={self.model.x_input: x_nat,
                                              self.model.y_input: y})
    best_loss = np.expand_dims(best_loss, 1)
    for it in range(iters):
      if self.rand:
        x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      else:
        x = np.copy(x_nat)

      for i in range(self.k):
        grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                              self.model.y_input: y})

        if self.norm == 'inf':
            x += self.a * np.sign(grad) # update
            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) # project
        else: # two
            # update
            grad_norm = self._l2_norm(grad)
            grad_norm = grad_norm.clip(1e-8, np.inf) # protect against zero grad
            x += self.a * grad / grad_norm
            # project
            dx = x - x_nat
            dx_norm = self._l2_norm(dx)
            dx_final_norm = dx_norm.clip(0, self.epsilon)
            x = x_nat + dx_final_norm * dx / dx_norm
        x = np.clip(x, 0, 1)

      new_loss = sess.run(self.loss, feed_dict={self.model.x_input: x,
                                                self.model.y_input: y})
      new_loss = np.expand_dims(new_loss, 1)

      old_mask = np.where(best_loss >= new_loss, 1, 0)
      new_mask = np.where(new_loss > best_loss, 1, 0)
      best_x = old_mask * best_x + new_mask * x
      best_loss = old_mask * best_loss + new_mask * new_loss

    return best_x

  @staticmethod
  def _l2_norm(batch):
      return np.sqrt((batch ** 2).sum(axis=1, keepdims=True))


  @staticmethod
  def _l2_ball_sample(radius, shape):
      direction = np.random.normal(0.0, 1.0, shape)
      direction = direction / __class__._l2_norm(direction)
      dimension = np.prod(shape[1:]) # product of non-batch dimensions
      magnitude = radius * np.random.uniform(0, 1, (shape[0],1)) ** (1 / dimension)
      return magnitude * direction


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open(args.config) as config_file:
    config = json.load(config_file)

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  model = Model()
  attack = PGDAttack(model,
                     config['norm'],
                     config['epsilon'],
                     config['k'],
                     config['a'],
                     config['random_start'],
                     config['loss_func'])
  saver = tf.train.Saver()

  mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = mnist.test.images[bstart:bend, :]
      y_batch = mnist.test.labels[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess, iters=config['random_iters'])

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))
