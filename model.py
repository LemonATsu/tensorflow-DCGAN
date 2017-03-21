import numpy as np
import tensorflow as tf
from utils import *
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

class DCGAN(object):

    def __init__(self, sess, z_dim=100, batch_size=128, img_size=64, n_channel=1):

        self.sess = sess
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.img_size = [img_size, img_size] # assume that w = h
        self.n_channel = n_channel

    def build_graph(self):
        self.z = tf.placeholder(tf.float32, [None, z_dim])

        self.real_sample = tf.placeholder(tf.float32, [batch_size] + img_size + [self.n_channel],
                                            name='real_sample')
        self.fake_sample = tf.placeholder(tf.float32, [batch_size] + img_size + [self.n_channel],
                                            name='fake_sample')

        self.G = generator(z, self.img_size, name='G')
        self.D_real = discriminator(self.real_sample, self.img_size, name='D')
        self.D_fake = discriminator(self.fake_sample, self.img_size, name='D')
        # predicted label from discriminator
        D_real = tf.nn.sigmoid(self.D_real)
        D_fake = tf.nn.sigmoid(self.D_fake)


def generator(z, img_size=64, name='gen', reuse=True):

    with tf.variable_scope(name, reuse=reuse):
        l = linear(z, img_size*16*4*4, name='g_lin1')
        l = tf.reshape(l, [-1, 4, 4, img_size*16])
        batch_size = tf.shape(l)[0]
        l = tf.nn.relu(batch_norm(l))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 8,  8, img_size*8], name='g_deconv1')))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 16, 16, img_size*4], name='g_deconv2')))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 32, 32, img_size*2], name='g_deconv3')))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 64, 64, 3], name='g_deconv4')))

    return tf.nn.tanh(l)


def discriminator(input_tensor, input_size=[64, 64], output_size=1, name='discr', reuse=True):

    nf = input_size[0]
    with tf.variable_scope(name, reuse=reuse):
        l = tf.reshape(input_tensor, [-1] + input_size + [1]) # [-1, 64, 64, 1] by defalut
        l = lrelu(conv2d(l, nf, name='d_conv1'))
        l = lrelu(batch_norm(conv2d(l, nf*2, name='d_conv2')))
        l = lrelu(batch_norm(conv2d(l, nf*4, name='d_conv3')))
        l = lrelu(batch_norm(conv2d(l, nf*8, name='d_conv4')))
        l = flatten(l)
        l = linear(l, output_size, name='d_lin1')

    return l

if __name__ == '__main__':
    dummy = tf.placeholder(tf.float32, [10, 28, 28, 1])
    d = discriminator(dummy, [28, 28])
    g = generator(dummy)
