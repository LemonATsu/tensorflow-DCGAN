import tensorflow as tf

def deconv2d(input_tensor, output_dim, k=5, s=2, stddev=.02, name='deconv2d'):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, output_dim[-1], input_tensor.get_shape()[-1]],
                                initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_tensor, w, output_shape=output_dim, strides=[1, s, s, 1])
        b = tf.get_variable('b', [output_dim[-1]], initializer=tf.constant_initializer(.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, b), deconv.get_shape())

    return deconv

def conv2d(input_tensor, output_dim, k=5, s=2, stddev=.02, name='conv2d'):

    with tf.variable_scope(name):
        w = tf.get_variable('w', [k, k, input_tensor.get_shape()[-1], output_dim],
                                initializer=tf.truncated_normal_initializer(stddev=stddev))

        conv = tf.nn.conv2d(input_tensor, w, strides=[1, s, s, 1], padding='VALID')
        b = tf.get_variable('b', [output_dim], initializer=tf.constant_initializer(.0))
        conv = tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

    return conv

def linear(input_tensor, output_size, stddev=.02, name='lin'):

    shape = input_tensor.get_shape().as_list()
    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], output_size], tf.float32,
                tf.random_normal_initializer(stddev=stddev))

        b = tf.get_variable('b', [output_size], initializer=tf.constant_initializer(.0))

    return tf.matmul(input_tensor, w) + b

