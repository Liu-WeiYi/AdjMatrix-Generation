#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  定义一些常用函数
  Created: 04/08/17
"""
import tensorflow as tf
import tensorflow.contrib.slim as slim # 一个tensorflow的小库~~

# ------------------------------
# show_all_variables()
# @purpose: 展示TF中的所有变量
# ------------------------------
def show_all_variables():
    models_vars = tf.trainable_variables()
    # Prints the names and shapes of the variables
    slim.model_analyzer.analyze_vars(models_vars,print_info=True)

# ------------------------------------------------------
# class batch_norm(object)
# @purpose: 对传入的batch进行 norm 操作 --- based on DCGAN
# ------------------------------------------------------
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
        with tf.variable_scope(name_or_scope=name):
            self.epsilon = epsilon
            self.momentum = momentum
            self.name = name

    def __call__(self, batch, train=True):
        return tf.contrib.layers.batch_norm(batch,
                                            decay=self.momentum,
                                            updates_collections=None,
                                            epsilon=self.epsilon,
                                            scale=True,
                                            is_training=train,
                                            scope=self.name
                                        )

# ---------------------------------------------------------------------------------------------
# linear
# @purpose: 对传入的input进行 线性化 操作 --- based on DCGAN
# @input:
#        input_ : 输入向量
#        output_size : 输出向量的shape
#        scope : 名称
#        stddev : 用于高斯矩阵的标准差是多少
#        bias_start : 规定 bias 的大小
#        with_w : False ~~ FALSE表示直接返回Wx+B，如果为TRUE，则返回Wx+B的同时，再返回 matrix 和 bias
# ---------------------------------------------------------------------------------------------
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                    tf.random_normal_initializer(stddev=stddev))# 声明1010 * 1024矩阵
        bias = tf.get_variable("bias", [output_size],
        initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias

# ------------------------------
# conv_cond_concat()
# @purpose: 合并两个Tensors --- based on DCGAN
# ------------------------------
def conv_cond_concat(x, y):
    """Concatenate conditioning vector on feature map axis."""
    x_shapes = x.get_shape()
    y_shapes = y.get_shape()
    return tf.concat([
        x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)

# ---------------------------------------------------------------------------------------------
# deconv2d
# @purpose: 对传入的input进行 反卷积 操作 --- based on DCGAN
# @input:
#        input_ : 输入向量
#        output_size : 输出向量的shape
#        k_h : filter 高度
#        k_w : filter 宽度
#        d_h : strides 高度
#        d_w : strides 宽度
#        stddev : 用于高斯矩阵的标准差是多少
#        name : 名称
#        with_w : False ~~ FALSE表示直接返回Wx+B，如果为TRUE，则返回Wx+B的同时，再返回 matrix 和 bias
# ---------------------------------------------------------------------------------------------
def deconv2d(input_, output_shape,
            k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
            name="deconv2d", with_w=False):
  with tf.variable_scope(name):
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
              initializer=tf.random_normal_initializer(stddev=stddev))

    try:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    # Support for verisons of TensorFlow before 0.7.0
    except AttributeError:
      deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if with_w:
      return deconv, w, biases
    else:
      return deconv