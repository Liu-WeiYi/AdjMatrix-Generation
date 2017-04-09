#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  model (based on DCGAN)
  Created: 04/08/17
"""
import numpy as np
import tensorflow as tf

from utils import *

debugFlag = True

class adj_Matrix_Generator(object):
    """
    主类
    """
    def __init__(self,
                    sess, dataset_name,
                    epoch=10, learning_rate=0.0002, Momentum=0.5,
                    train_size=np.inf, batch_size=64,
                    generatorFilter=64, discriminatorFilter=64,
                    generatorFC=1024, discriminatorFC=1024,
                    inputMat_H=100, inputMat_W=100, outputMat_H=100, outputMat_W=100,
                    OutDegree_Length=100, InitGen_Length=100,
                    inputPartitionDIR="./graphs", checkpointDIR="./checkpoint", sampleDIR="./samples"
        ):
        """
        @purpose
            set all hyperparameters
        @inputs:
            sess:               Current Tensorflow Session
            dataset_name:       Current Dataset Name
            epoch:              Epochs Number for Whole Datsets [10]
            learning_rate:      Init Learning Rate for Adam [0.0002]
            Momentum:           采用Adam算法时Momentum的值 [0.5]
            train_size:         训练的多少 [np.inf]
            batch_size:         每一次读入的adj-matrix数量 [64]
            generatorFilter:    生成器初始的filter数值 [64] --- 注: 算法中每一层的filter都设置为该值的2倍 based on DCGAN
            discriminatorFilter:判别器初始的filter数值 [64] --- 注: 算法中每一层的filter都设置为该值的2倍 based on DCGAN
            generatorFC:        生成器的全连接层的神经元个数 [1024]
            discriminatorFC:    判别器的全连接层的神经元个数 [1024]
            inputMat_H:        输入邻接矩阵的Height [100]
            inputMat_W:         输入邻接矩阵的Width [100]
            outputMat_H:        输出邻接矩阵的Height [100]
            outputMat_W:        输出邻接矩阵的Width [100]
            OutDegree_Length:   当前AdjMatrix的出度向量长度 [100]  --- 相当于DCGAN中的 y_lim
            InitGen_Length:     初始化GEN时的输入向量大小 [100] --- 相当于DCGAN中的 z_lim
            inputPartitionDIR:  分割后的矩阵的存档点 [./graphs]
            checkpointDIR:      存档点 地址 [./checkpoint]
            sampleDIR:          输出图像地址 [./samples]
        """
        # 参数初始化
        self.sess                = sess
        self.dataset_name        = dataset_name
        self.epoch               = epoch
        self.learning_rate       = learning_rate
        self.Momentum            = Momentum
        self.train_size          = train_size
        self.batch_size          = batch_size
        self.generatorFilter     = generatorFilter
        self.discriminatorFilter = discriminatorFilter
        self.generatorFC         = generatorFC
        self.discriminatorFC     = discriminatorFC
        """
        注意，这里的inputMat 和 outputMat 均指向的是 生成器 千万不要搞错了!
        """
        self.inputMat_H          = inputMat_H
        self.inputMat_W          = inputMat_W
        self.outputMat_H         = outputMat_H
        self.outputMat_W         = outputMat_W
        self.OutDegreeLength     = OutDegree_Length
        self.InitSampleLength    = InitGen_Length
        self.inputPartitionDIR   = inputPartitionDIR
        self.checkpointDIR       = checkpointDIR
        self.sampleDIR           = sampleDIR
        """
        因为后面多个地方都需要用到batch_norm操作，因此在这里事先进行定义
        """
        self.generator_batch_norm_0 = batch_norm(name='g_bn0')
        self.generator_batch_norm_1 = batch_norm(name='g_bn1')
        self.generator_batch_norm_2 = batch_norm(name='g_bn2')

        self.discriminator_batch_norm1 = batch_norm(name="d_bn1")
        self.discriminator_batch_norm2 = batch_norm(name="d_bn2")

        # 创建GAN --- based on DCGAN
        self.modelConstrunction()

    def modelConstrunction(self):
        # 1. place holder 装输入的变量
        self.OutDegreeVector = tf.placeholder(tf.float32, shape=[self.batch_size, self.OutDegreeLength], name="Out_Degree_Vector")
        self.inputMat = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputMat_H, self.inputMat_W, 1], name="Real_Input_Adj_Matrix")
        self.sampleInput = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputMat_H, self.inputMat_W, 1], name="Sample_Input_Adj_Matrix")
        self.InitSampleVector = tf.placeholder(tf.float32, shape=[None, self.InitSampleLength], name="GEN_Input")

        # 2. generator
        self.Generator = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector)

        # 4. 放入TensorBoard中
        self.InitSampleVector_sum = tf.summary.histogram(name="GEN Input",values=self.InitSampleVector)

    def __generator(self, InitInputSampleVector, Input_OutDegreeVector):
        """
        @porpose 实现 生成器
        @生成器架构: ---> 该生成器 是一个 反卷积网络
                RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu
                        -> generatorFC -> batch_norm_1 -> relu
                            -> batch_norm -> deconv -> batch_norm_2 -> relu
                                -> deconv -> batch_norm3 -> sigmoid -> generated_Adj-Matrix

        @InitInputSampleVector : 放入 生成器 中的 一个 随机的 初始向量
        @Input_OutDegreeVector : 放入 生成器 中的 一个 Adj-Matrix的出度元素列表向量
        """
        with tf.variable_scope("generator") as scope:
            # 1. 定义卷积层的图像尺寸
            Height, Width = self.outputMat_H, self.outputMat_W
            deconv1_Height, deconv1_Width = int(Height/2), int(Width/2)
            deconv2_Height, deconv2_Weight = int(deconv1_Height/2), int(deconv1_Width/2)

            #  2.  RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu := h0 -> h0+OutDegreeVector
            """p.s. 下面所有的所谓"向量"其实都是一个矩阵，其第一个维度就是batch_size"""
            ## 2.1 将随机 初始的向量 和 出度元素列表向量 进行拼接
            input = tf.concat([InitInputSampleVector, Input_OutDegreeVector], axis=1) # <- input 为一个 100+100 共 200维 的向量
            if debugFlag is True:
                print('input shape: ',input.shape) # (64, 200)
            ## 2.2 经过全连接层->进行向量化->通过激活函数relu
            h0=tf.nn.relu(
                self.generator_batch_norm_0(linear(input_=input, output_size=self.generatorFC, scope="g_h0_lin"))) # <- h0 为一个 1024维 的向量
            if debugFlag is True:
                print('h0 shape: ', h0.shape) # (64, 1024)
            ## 2.3 再将Input_OutDegreeVector放在后面
            h0 = tf.concat([h0, Input_OutDegreeVector], axis=1) # <- 添加h0 为一个 1024维 的向量+ 100 维的那个OutDegreeVector [此时就变成了 64 * 1124 的矩阵]
            if debugFlag is True:
                print('h0+outdegree shape: ', h0.shape) # (64, 1124)

            # 3. h0+OutDegreeVector -> generatorFC -> batch_norm_1 -> relu := h1 -> reshape(h1) -> h1+OutDegreeVector
            output_size = self.generatorFilter*2 * deconv2_Height * deconv2_Weight  # <- 进一步增大维度，此时的维度应该是原“第二次卷积”之后的维度。
                                                                                    #  --- self.generatorFilter*2 表示CNN网络所有的卷积filter多少
                                                                                    #  --- deconv2_Height * deconv2_Width 表示CNN网络第二次卷积之后一共的个数
                                                                                    #  在这里，有 deconv2_Height = deconv2_Width = 100/4 = 25
                                                                                    #           self.generatorFilter*2 = 64*2 = 128
                                                                                    #           表明，线性化的长度应该是 output_size = 128*25*25 = 80000
                                                                                    # ∴ h1.shape = [64, 80000]
            h1 = tf.nn.relu(
                self.generator_batch_norm_1(linear(input_=h0, output_size=output_size, scope="g_h1_lin"))
            )
            if debugFlag is True:
                print('h1 shape: ', h1.shape) # (64, 80000)
            #  由于此时h1为一个向量，所以下面将这个向量reshape成一个Tensor，以配合（第二次）卷积的结果
            h1 = tf.reshape(h1, [self.batch_size, deconv2_Height, deconv2_Weight, self.generatorFilter*2])
            if debugFlag is True:
                print('reshape h1: ', h1.shape) # (64, 25, 25, 128)
            # 此时将reshape好的h1与之前的OutDegreeVector进行叠加。
            # 这里需要注意的是，由于之前OutDegreeVector仅仅是一个 64*100 的向量，而这里 h1代表的是一个Tensor，所以应该先将OutDegreeVector扩展成一个Tensor，再进行叠加
            # 由于h1.shape = [64, 25, 25, 128], 所以对应的OutDegreeTensor应该是每一个都有~所以应该写成下面这个形式
            OutDegreeTensor = tf.reshape(self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('reshape OutDegreeVector: ', OutDegreeTensor.shape)
            # 将OutDegreeTensor放在后面
            h1 = conv_cond_concat(x=h1, y=OutDegreeTensor)
            if debugFlag is True:
                print('reshape h1 + OutDegreeVector: ', h1.shape) # (64, 25, 25, 128+100)

            # 4. h1+OutDegreeVector -> deconv2d -> batch_norm_2 -> relu := h2 -> h2+OutDegreeVector
            h2 = tf.nn.relu(
                self.generator_batch_norm_2(
                    deconv2d(input_=h1, output_shape=[self.batch_size, deconv1_Height, deconv1_Width, self.generatorFilter*2], name="g_h2")
                )
            )
            if debugFlag is True:
                print('deconv h2: ', h2.shape) # (64, 50, 50, 128)
            # 将OutDegreeTensor放在后面
            h2 = conv_cond_concat(x=h2, y=OutDegreeTensor)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape)

            # 5. h2+OutDegreeVector -> deconv2d -> sigmoid # 注意最后一层没有做batch_norm!!!
            h3 = tf.nn.sigmoid(
                deconv2d(input_=h2, output_shape=[self.batch_size, Height, Width, self.generatorFilter*2], name="g_h3")
            )
            if debugFlag is True:
                print('final layer h3 shape: ', h3.shape)

            return h3

