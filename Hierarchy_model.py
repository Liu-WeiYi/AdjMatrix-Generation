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
import Conditional_model as C_model

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
                inputPartitionDIR="facebook", checkpointDIR="condition_checkpoint", sampleDIR="condition_samples",reconstructDIR="reconstruction",
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
            OutDegree_Length:   当前 AdjMatrix的 出度向量长度，表示的是类别 [28]  --- 相当于DCGAN中的 y_lim [手写数字中的类别信息~]
            inputPartitionDIR:  分割后的矩阵的存档点 [facebook] --- 注: 最好与 dataset_name 保持一致，只不过这里指的是当前dataset_name所在的folder
            checkpointDIR:      存档点 地址 [condition_checkpoint]
            sampleDIR:          采样得到的网络 输出地址 [condition_samples]
            reconstructDIR:     重构网络的存放地址
            link_possibility:   重构网络时指定的 连接权重
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
        self.inputMat_H_list          = [info[1] for info in train_list]
        self.inputMat_W_list          = [info[1] for info in train_list]
        self.outputMat_H_list         = [info[1] for info in train_list]
        self.outputMat_W_list         = [info[1] for info in train_list]
        self.OutDegree_Length         = OutDegree_Length
        # 用作Generator的输入，生成 当前网络
        self.InitSampleLength_list    = [info[1] for info in train_list]
        # 指定 路径
        self.inputPartitionDIR   = inputPartitionDIR
        self.checkpointDIR       = checkpointDIR
        self.sampleDIR           = sampleDIR
        self.reconstructDIR      = reconstructDIR
        # 用作生成 拓扑结构 的方式
        self.link_possibility    = link_possibility

        # 构建 GAN~
        self.modelConstrunction()

        print('so far so good as;dfas;ofhasd;ifhas;oifaw;ofwqpofpwoif[qwfwqopfeqwofb')

    def modelConstrunction(self):
        print('\n============================================================================')
        print('Model Construction ...')
        print('============================================================================')

        self.reconstructNet_per_layer = []
        for layer_idx in range(len(self.trainable_data_size_list)):
            """训练每一个模型~"""
            model_name = "%s_Mat_%d_Trainable_%d"%(self.dataset_name, self.inputMat_H_list[layer_idx], self.trainable_data_size_list[layer_idx])
            if debugFlag is True:
                print('current model: ', model_name)
            # with tf.device('cpu:0'):
            with tf.Session() as sess:
                model = C_model.Condition_adjMatrix_Generator(
                    sess,dataset_name = self.dataset_name,
                    epoch=self.epoch,learning_rate=self.learning_rate,Momentum=self.Momentum,
                    batch_size=self.batch_size,
                    generatorFilter=self.generatorFilter,discriminatorFilter=self.discriminatorFilter,
                    generatorFC=self.generatorFC,discriminatorFC=self.discriminatorFC,
                    trainable_data_size=self.trainable_data_size_list[layer_idx],
                    inputMat_H=self.inputMat_H_list[layer_idx],
                    inputMat_W=self.inputMat_W_list[layer_idx],
                    outputMat_H=self.outputMat_H_list[layer_idx],
                    outputMat_W=self.outputMat_W_list[layer_idx],
                    OutDegree_Length=self.OutDegree_Length,
                    InitGen_Length=self.InitSampleLength_list[layer_idx],
                    inputPartitionDIR=self.inputPartitionDIR,
                    checkpointDIR=os.path.join(self.checkpointDIR, model_name),
                    sampleDIR=os.path.join(self.sampleDIR, model_name),
                    link_possibility=self.link_possibility
                )
                model.train()
                model.saveModel()
                re_Net = model.reconstructMat()

                self.reconstructNet_per_layer.append(re_Net)