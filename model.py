import numpy as np
import tensorflow as tf
from utils import *
from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.nn import sigmoid_cross_entropy_with_logits

class DCGAN(object):

    def __init__(self, sess, z_dim=100, batch_size=128, img_size=64, n_channel=1):

        self.sess = sess
        self.z_dim = z_dim
        self.batch_size = batch_size
        self.img_size = [img_size, img_size] # assume that w = h
        self.n_channel = n_channel

        self.build_graph()

    def build_graph(self):
        # input of generator, which should be some random number
        self.z = tf.placeholder(tf.float32, [None, z_dim])

        # setup real and fake input
        self.real_input = tf.placeholder(tf.float32, [batch_size] + img_size + [self.n_channel],
                                            name='real_sample')
        self.fake_input = tf.placeholder(tf.float32, [batch_size] + img_size + [self.n_channel],
                                            name='fake_sample')

        self.G = generator(z, self.img_size, self.n_channel, name='G')

        # use the same 'D' to discriminate generated data and real data
        self.D_real = discriminator(self.real_input, self.img_size, self.n_channel, name='D')
        self.D_fake = discriminator(self.G, self.img_size, self.n_channel, name='D')

        # predicted label from discriminator
        D_real = tf.nn.sigmoid(self.D_real)
        D_fake = tf.nn.sigmoid(self.D_fake)

        # set up the loss function of discriminator
        D_loss_real = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_real, tf.ones_like(D_real)))
        D_loss_fake = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.D_fake, tf.zeros_like(D_fake)))
        self.G_loss = tf.reduce_mean(
                sigmoid_cross_entropy_with_logits(self.G, tf.ones_like(D_fake)))

        self.D_loss = D_loss_real + D_loss_fake

        # get trainable vars of each network
        all_vars = tf.trainable_variables()
        self.D_vars = [var for var in all_vars if 'd_' in var.name]
        self.G_vars = [var for var in all_vars if 'g_' in var.name]

        # add to monitor
        tf.scalar_summary('g_loss', G_loss)
        tf.scalar_summary('d_loss', D_loss)
        tf.image_summary('G', self.G)

        self.saver = tf.train.Saver()

    def train(self, cfg):
        # set up optimizer
        g_opt = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1)
                    .minimize(self.G_loss, var_list=self.G_vars)
        d_opt = tf.train.AdamOptimizer(cfg.lr, beta1=cfg.beta1)
                    .minimize(self.D_loss, var_list=self.D_vars)

        tf.global_variables_initializer.run()




def generator(z, img_size=[64, 64], n_channel=1, name='gen', reuse=True):

    # generator architectures :
    # each deconv layer should be followd by batchnorm and Relu
    size = img_size[0]
    with tf.variable_scope(name, reuse=reuse):
        l = linear(z, size*16*4*4, name='g_lin1')
        l = tf.reshape(l, [-1, 4, 4, size*16])
        batch_size = tf.shape(l)[0]
        l = tf.nn.relu(batch_norm(l))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 8,  8, size*8], name='g_deconv1')))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 16, 16, size*4], name='g_deconv2')))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size, 32, 32, size*2], name='g_deconv3')))
        l = tf.nn.relu(batch_norm(deconv2d(l, [batch_size] + img_size + [n_channel], name='g_deconv4')))

    return tf.nn.tanh(l)


def discriminator(input_tensor, input_size=[64, 64], n_channel=1, output_size=1, name='discr', reuse=True):

    # discriminator architectures :
    # each conv layer should be followed by batchnorm and leakyRelu,
    # except for the first conv, which does not need to apply batchnorm
    nf = input_size[0]
    with tf.variable_scope(name, reuse=reuse):
        l = tf.reshape(input_tensor, [-1] + input_size + [n_channel]) # [-1, 64, 64, 1] by defalut
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
