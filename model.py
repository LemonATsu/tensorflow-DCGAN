import numpy as np
import tensorflow as tf
from utils import *
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def generator(z, img_size=64):

    l = linear(z, img_size*16*4*4, name='g_lin1')
    l = tf.reshape(l, [-1, 4, 4, img_size*16])
    batch_size = tf.shape(l)[0]
    l = tf.nn.relu(batch_norm(l))
    l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 8,  8, img_size*8], name='g_deconv1')))
    l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 16, 16, img_size*4], name='g_deconv2')))
    l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 32, 32, img_size*2], name='g_deconv3')))
    l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 64, 64, 3], name='g_deconv4')))

    return tf.nn.tanh(l)


def discriminator(input_tensor, input_size=[64, 64], output_size=1):

    nf = input_size[0]
    l = tf.reshape(input_tensor, [-1] + input_size + [1]) # [-1, 64, 64, 1] by defalut
    l = lrelu(conv2d(l, nf, k=5, s=2, name='d_conv1'))
    l = lrelu(batch_norm(conv2d(l, nf*2, k=5, s=2, name='d_conv2')))
    l = lrelu(batch_norm(conv2d(l, nf*4, k=5, s=2, name='d_conv3')))
    l = lrelu(batch_norm(conv2d(l, nf*8, k=5, s=2, name='d_conv4')))
    l = flatten(l)
    l = linear(l, output_size, name='d_lin1')

    return l

if __name__ == '__main__':
    d = discriminator(dummy)
    g = generator(dummy)
