#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Hierarchy model (based on Condition Model & DCGAN)
  Created:  04/13/17
"""
import numpy as np
import tensorflow as tf
import time
import os
import pickle
import glob
import math

from utils import *

debugFlag = True

class Hierarchy_adjMatrix_Generator(object):
    """
    @purpose: 随便给定一个网络，生成具有对应特性的网络
    """
    def __init__(self,
                 sess, dataset_name,
                 epoch=10, learning_rate=0.0002, Momentum=0.5,
                 batch_size=10,
                 generatorFilter=50, discriminatorFilter=50,
                 generatorFC=1024, discriminatorFC=1024,
                 training_info_dir="facebook_partition_info.pickle",
                 OutDegree_Length=1,
                 inputPartitionDIR="facebook", checkpointDIR="condition_checkpoint", sampleDIR="condition_samples",
                 link_possibility=0.5
                 ):
        """
        @purpose
            set all hyperparameters
        @inputs:
            sess:               Current Tensorflow Session
            dataset_name:       Current Dataset Name
            epoch:              Epochs Number for Whole Datsets [20]
            learning_rate:      Init Learning Rate for Adam [0.0002]
            Momentum:           采用ADAM算法时所需要的Momentum的值 [0.5] --- based on DCGAN
            batch_size:         每一批读取的adj-mat大小 [10]
            generatorFilter:    生成器 初始的filter数值 [50] --- 注: 算法中每一层的filter都设置为该值的2倍 based on DCGAN
            discriminatorFilter:判别器 初始的filter数值 [50] --- 注: 算法中每一层的filter都设置为该值的2倍 based on DCGAN
            generatorFC:        生成器的全连接层的神经元个数 [1024]
            discriminatorFC:    判别器的全连接层的神经元个数 [1024]
            training_info_dir:  [trainable_data_size, inputMatSize] 所在位置
            OutDegree_Length:   当前 AdjMatrix的 出度向量长度，最好与inputMat保持一致 [28]  --- 相当于DCGAN中的 y_lim
            inputPartitionDIR:  分割后的矩阵的存档点 [facebook] --- 注: 最好与 dataset_name 保持一致，只不过这里指的是当前dataset_name所在的folder
            checkpointDIR:      存档点 地址 [condition_checkpoint]
            sampleDIR:          采样得到的网络 输出地址 [condition_samples]
        """
        # GAN 参数初始化
        self.sess                = sess
        self.dataset_name        = dataset_name
        self.epoch               = epoch
        self.learning_rate       = learning_rate
        self.Momentum            = Momentum
        self.batch_size          = batch_size
        self.generatorFilter     = generatorFilter
        self.discriminatorFilter = discriminatorFilter
        self.generatorFC         = generatorFC
        self.discriminatorFC     = discriminatorFC
        # input / output size 初始化
        train_list = pickle.load(open(training_info_dir,'rb'))
        self.trainable_data_size_list = [info[0] for info in train_list]
        self.inputMat_H_lise          = [info[1] for info in train_list]
        self.inputMat_W_lise          = [info[1] for info in train_list]
        self.outputMat_H_lise         = [info[1] for info in train_list]
        self.outputMat_W_lise         = [info[1] for info in train_list]
        self.OutDegreeLength          = OutDegree_Length
        # 用作Generator的输入，生成 当前网络
        self.InitSampleLength_lise    = [info[1] for info in train_list]
        # 指定 路径
        self.inputPartitionDIR   = inputPartitionDIR
        self.checkpointDIR       = checkpointDIR
        self.sampleDIR           = sampleDIR
        # 用作生成 拓扑结构 的方式
        self.link_possibility    = link_possibility

        """
        因为后面多个地方都需要用到 batch_norm 操作，因此在这里事先进行定义 --- based on DCGAN
        """
        self.generator_batch_norm_0 = batch_norm(name='g_bn0')
        self.generator_batch_norm_1 = batch_norm(name='g_bn1')
        self.generator_batch_norm_2 = batch_norm(name='g_bn2')

        self.discriminator_batch_norm1 = batch_norm(name="d_bn1")
        self.discriminator_batch_norm2 = batch_norm(name="d_bn2")

        # 构建 GAN~
        # self.modelConstrunction()