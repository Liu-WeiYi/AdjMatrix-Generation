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
from Conditional_Topology_MAIN import graph2Adj, generate_graph

debugFlag = True

class Hierarchy_adjMatrix_Generator(object):
    """
    @purpose: 随便给定一个网络，生成具有对应特性的网络
    """
    def __init__(self,
                sess, dataset_name, permutation_num,
                epoch=10, learning_rate=0.0002, Momentum=0.5,
                batch_size=10,
                generatorFilter=50, discriminatorFilter=50,
                generatorFC=1024, discriminatorFC=1024,
                training_info_dir="facebook_partition_info.pickle",
                OutDegree_Length=1,
                inputPartitionDIR="facebook", checkpointDIR="condition_checkpoint", sampleDIR="condition_samples",reconstructDIR="reconstruction",
                link_possibility=0.5,
                trainedFlag=False
            ):
        """
        @purpose
            set all hyperparameters
        @inputs:
            sess:               Current Tensorflow Session
            dataset_name:       Current Dataset Name
            permutation_step:   Permutate Num
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
            trainedFlag:        是否需要对每一层进行训练. 当为FALSE表示需要训练，TRUE表示不需要训练 [False]
        """
        # GAN 参数初始化
        self.sess                = sess
        self.dataset_name        = dataset_name
        self.permutation_num     = permutation_num
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
        if trainedFlag is False:
            self.per_layer_modelConstrunction()
            show_all_variables() # TF中的所有变量
            print('Trained Layers Process DOWN...')

    def per_layer_modelConstrunction(self):
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
                re_Net = model.reconstructMat(type="Hierarchy")

                self.reconstructNet_per_layer.append(re_Net)

    def modelConstruction(self):
        # step.0 生成Hierarchy GAN 的Adj 以及 原始数据的 GAN
        if not os.path.exists('%s_adjs.pickle'%self.dataset_name):
            # 1. 读取trained 后的每一层的数据, 并生成 保存于trained_graph_list中
            trained_layer_path = os.path.join(self.reconstructDIR,self.dataset_name,"Hierarchy",'')
            trained_graph_adj_list = []
            paths = glob.glob(trained_layer_path+"%s_*.nxgraph"%self.dataset_name)
            if debugFlag is True:
                print('all trained layer paths: ', paths)
            for path in paths:
                graph = pickle.load(open(path,'rb'))
                if debugFlag is True:
                    print('trained graph size: ',len(graph.nodes()))
                adj = graph2Adj(graph, max_size = -1)
                if debugFlag is True:
                    print('current adj shape: ', adj.shape)
                trained_graph_adj_list.append(adj)

            # 2. 读取原始网络，并生成对应的adj
            original_graph_path = os.path.join("data", self.dataset_name, '')
            origin_graph = generate_graph(original_graph_path,self.dataset_name,-1)
            if debugFlag is True:
                print('original graph size: ',len(origin_graph.nodes()))
            origin_adj = graph2Adj(origin_graph,max_size=-1)
            if debugFlag is True:
                print('original adj shape: ', origin_adj.shape)

            pickle.dump([trained_graph_adj_list,origin_adj],open('%s_adjs.pickle'%self.dataset_name,'wb'))
        else:
            [trained_graph_adj_list,origin_adj] = pickle.load(open('%s_adjs.pickle'%self.dataset_name,'rb'))
            if debugFlag is True:
                for i in trained_graph_adj_list:
                    print('trained graph ajd shape :', i.shape)
                print('original adj shape: ', origin_adj.shape)

        """ add 2017.04.18 """
        # ==============================================
        # 1. 将trained_graph_list 改成也需要考虑原网络的情况
        # ==============================================
        trained_graph_adj_list.append(origin_adj)

        # ==============================================
        # 2. Permute Adj to generate more adjs
        # ==============================================
        permute_number = self.permutation_num
        if debugFlag is True:
            print("permuting each adj for %d times"%permute_number)
        trained_graph_adj_list = permute_adjs(trained_graph_adj_list,permute_number)
        if debugFlag is True:
            print('permuted adjs number: ', len(trained_graph_adj_list))
        """ end add """

        # step.1 创建Weight~
        self.trained_graph_weight_list = []
        self.layer_weight_list = [] # for Hierarchy GAN~ 😀
        count = 0
        for adj in trained_graph_adj_list:
            """每一个邻接矩阵 生成不一样的权重"""
            layer_weight = tf.Variable(tf.random_uniform([1],minval=0,maxval=1),name="weight_%d"%count)
            self.layer_weight_list.append(layer_weight)
            adj_layer_weight = layer_weight*tf.ones(shape=adj.shape) # 扩展到每一个维度上~
            self.trained_graph_weight_list.append(adj_layer_weight)

            count += 1

        # 创建bias
        self.bias = tf.Variable(tf.random_uniform([1],minval=0,maxval=1),name="bias")

        # step.2 logit
        tmp = [self.trained_graph_weight_list[idx]*trained_graph_adj_list[idx] for idx in range(len(self.trained_graph_weight_list))]
        self.logits = tf.add(tf.add_n(tmp,name="Layered_results"),self.bias)

        # step.3 loss
        origin_adj = tf.to_float(origin_adj)
        """算L1距离"""
        # self.loss = tf.reduce_mean(tf.square(origin_adj-self.logits)) # L2-norm
        # self.loss = tf.reduce_mean(tf.abs(origin_adj-self.logits))      # L1-norm
        """算度之间的差别"""
        # zeros = tf.zeros(shape=origin_adj.shape)
        # self.loss = tf.reduce_sum(tf.to_float(tf.where(tf.not_equal(origin_adj-self.logits,zeros))))
        """尝试算两个度分布之间的KL距离"""
        # self.loss = tf.contrib.distributions.kl(tf.reduce_sum(origin_adj,1), tf.reduce_sum(adj,1))
        self.logits = self.logits + 0.000001 * tf.ones(shape=origin_adj.shape) # 保证分母不为0
        y = origin_adj/self.logits

        self.loss = tf.abs(tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)))

        """learning rate decay... from TF_API"""
        start_learning_rate = 0.1
        global_step = tf.Variable(0, trainable=False)
        decay_step = 1000
        decay_rate = 0.96
        learning_rate = tf.train.exponential_decay(learning_rate=start_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=decay_step,
                                                    decay_rate=decay_rate)

        # step.4 optimizer
        # self.opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self, training_step=10000):
        """
        @input training_step 训练所需步骤
        @return weight 列表， reconstructed adj
        """
        tf.global_variables_initializer().run()

        for step in range(training_step):
            self.sess.run(self.opt)
            info = ''
            for idx in range(len(self.layer_weight_list)):
                tmp = 'W_%d: [%.4f] | '%(idx, self.sess.run(self.layer_weight_list[idx]))
                info += tmp
            info += " bias: [%.4f]"%self.sess.run(self.bias)
            if len(self.layer_weight_list) <= 4:
                print('step: [%d]/[%d], loss value: %.4f'%(step+1, training_step, self.sess.run(self.loss)), info)
            else:
                print('step: [%d]/[%d], loss value: %.4f'%(step+1, training_step, self.sess.run(self.loss)))

            if self.sess.run(self.loss) <= 0.01:
                break

        weight_list = [self.sess.run(self.layer_weight_list[idx]) for idx in range(len(self.layer_weight_list))]
        tmp_re_adj_raw = self.sess.run(self.logits)
        # meanValue = tf.reduce_mean(tmp_re_adj_raw)
        # tmp_re_adj_raw = tmp_re_adj_raw - meanValue*tf.ones(shape = tmp_re_adj_raw.shape)
        # 先将 原矩阵 正则化，使之投影到 [0, 1]空间中
        maxValue = tf.reduce_max(tmp_re_adj_raw)
        tmp_re_adj_raw_norm = tmp_re_adj_raw/maxValue
        # 再平移均值于-meanValue处，这是为了契合 sigmoid函数的图像特征.
        """
        因为 sigmoid(X≥0) ≥ 0.5, 而tmp_re_adj_raw_norm中的值均大于0,
        所以 如果不进行平移，会导致通过sigmoid映射之后的所有节点的值都在0.5以上
        """
        tmp_re_adj_raw = tmp_re_adj_raw_norm - 0.5*tf.ones(shape = tmp_re_adj_raw.shape)

        reconstructed_Adj = tf.sigmoid(tmp_re_adj_raw)
        return weight_list, reconstructed_Adj.eval()


