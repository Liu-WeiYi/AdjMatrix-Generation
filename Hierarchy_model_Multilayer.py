#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Multilayer Hierarchy model (based on Hierarchy_model.py)
  Created:  04/22/17
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
if debugFlag is True:
    import time

class Multilayer_Hiearchy_adjMatrix_Generator(object):
    """
    @purpose: 根据多个网络的特性，生成一个自定义大小的网络~
    """
    def __init__(self,sess, datasets_list, desired_size, reconstructDIR):
        """
        @purpose
            set all hyperparameters
        @inputs:
            sess:               Current Tensorflow Session
            dataset_name:       Current Dataset Name
            desired_size:       desired output reconstructed graph size
            reconstructDIR:     reconstructed graph's dir
        """
        # 初始化~
        self.sess           = sess
        self.datasets_list  = datasets_list
        self.desired_size   = desired_size
        self.reconstructDIR = reconstructDIR

    def modelConstruction(self):
        # step.0 读入/写入 训练数据 & 真实数据
        trained_graph_adj_list = []
        origin_adj_list = []

        for dataset in self.datasets_list:
            """是否存在 pickle 文件"""
            if not os.path.exists('%s_adjs.pickle'%dataset):
                # 1. 获取所有trained 数据, 并存储在trained_graph_list?
                current_trained_graph_adj_list = []
                trained_layer_path = os.path.join(self.reconstructDIR,dataset,"Hierarchy",'')
                paths = glob.glob(trained_layer_path+"%s_*.nxgraph"%dataset)
                if debugFlag is True:
                    print('all trained layer paths: ', paths)
                for path in paths:
                    graph = pickle.load(open(path,'rb'))
                    if debugFlag is True:
                        print('trained graph size: ',len(graph.nodes()))
                    adj = graph2Adj(graph, max_size = -1)
                    if debugFlag is True:
                        print('current adj shape: ', adj.shape)
                    current_trained_graph_adj_list.append(adj)

                trained_graph_adj_list+=current_trained_graph_adj_list

                # 2. 获取所有真实adj
                original_graph_path = os.path.join("data", dataset, '')
                origin_graph = generate_graph(original_graph_path,dataset,-1)
                if debugFlag is True:
                    print('original graph size: ',len(origin_graph.nodes()))
                origin_adj = graph2Adj(origin_graph,max_size=-1)
                if debugFlag is True:
                    print('original adj shape: ', origin_adj.shape)

                origin_adj_list.append(origin_adj)

                pickle.dump([current_trained_graph_adj_list,origin_adj],open('%s_adjs.pickle'%dataset,'wb'))

            else:
                [current_trained_graph_adj_list,current_origin_adj] = pickle.load(open('%s_adjs.pickle'%dataset,'rb'))
                if debugFlag is True:
                    for i in current_trained_graph_adj_list:
                        print('trained graph ajd shape :', i.shape)
                    print('original adj shape: ', current_origin_adj.shape)
                trained_graph_adj_list += current_trained_graph_adj_list
                origin_adj_list.append(current_origin_adj)

        if debugFlag is True:
            print('total trained graph adj length: ', len(trained_graph_adj_list))
            print('total origin graph adj length: ', len(origin_adj_list))

        # ===================================================
        # 1. 初始化 Weight 以及 Bias
        # ===================================================
        trained_graph_adj_list+= origin_adj_list

        ## step.1 init Weight~
        trained_graph_weight_list = []
        self.layer_weight_list = [] # for Hierarchy GAN~ ??
        count = 0
        for adj in trained_graph_adj_list:
            """ 对于每一个adj, 给出 weight """
            layer_weight = tf.Variable(tf.random_uniform([1],minval=0,maxval=1),name="weight_%d"%count)
            self.layer_weight_list.append(layer_weight)
            trained_graph_weight_list.append(layer_weight)

            count += 1

        ## step.2 init bias
        self.bias = tf.Variable(tf.random_uniform([1],minval=0,maxval=1),name="bias")

        # ===================================================
        # 2. 将所有重构的网络 map 到 self.desired_size*self.desired_size 上
        # 依据: re_adj = w1*g1+w2*g2+... +w*g_real + bias
        # ===================================================

        # step.1 map g1 -> g_desired_size
        tmp = []
        for idx in range(len(trained_graph_weight_list)):
            # 给定每一个weight
            adj_weight = trained_graph_weight_list[idx]
            adj = trained_graph_adj_list[idx]
            current_adj = adj_weight*adj
            # # 投影到 desired_size 上
            # linear_current_adj = tf.reshape(current_adj,[-1]) # flatten matrix~
            # linear_current_adj = tf.expand_dims(linear_current_adj,axis=-1)
            # linear_length = linear_current_adj.shape.as_list()[0]
            # """Y=AX"""
            # TransformM = tf.truncated_normal(shape=[self.desired_size**2, linear_length],mean=0.0,stddev=1.0)
            # reshape_linear_current_adj = tf.matmul(TransformM,linear_current_adj)
            # reshape_adj = tf.reshape(reshape_linear_current_adj, shape=[self.desired_size, self.desired_size])
            reshape_adj = self.__reshapeAdj(current_adj,self.desired_size)
            tmp.append(reshape_adj)

        ## step.2 w1*g1+w2*g2+... +wn*gn + w*g_real + bias~~
        ## 重构具有期望graph_size 的 graph
        self.logits = tf.add_n(tmp,name="Layered_results")+self.bias

        ## step.3 抽取重构矩阵的 degree_distribution self.logits_degree
        self.logits_degree = tf.reduce_sum(self.logits,1)
        """ 注意, 这里 degree distribution 是进行排序了的 !! """
        self.logits_degree,_ = tf.nn.top_k(self.logits_degree, k=self.logits_degree.get_shape().as_list()[0])

        # ===================================================
        # 3. 投影所有真实网络到预期网络上 self.desired_size*self.desired_size ?
        # ===================================================

        ## step.1 投影 g_real -> g_real_desired_size
        """注意, 这里的每一个真实网络应该对应一个 degree distribution, 而不应该像重构网络那样，所有的对应一个degree distribution"""
        origin_adjs_degree = []
        for adj in origin_adj_list:
            # 1. reshape original adj
            adj = tf.to_float(adj)
            # linear_adj = tf.reshape(adj,[-1])
            # linear_adj = tf.expand_dims(linear_adj,axis=-1)
            # linear_length = linear_adj.get_shape().as_list()[0]
            # TransformM = tf.truncated_normal(shape=[self.desired_size**2, linear_length],mean=0.0,stddev=1.0)
            # reshape_linear_adj = tf.matmul(TransformM, linear_adj)
            # reshape_adj = tf.reshape(reshape_linear_adj, shape=[self.desired_size, self.desired_size])
            reshape_adj = self.__reshapeAdj(adj, self.desired_size)
            # 2. get degree value
            reshape_adj_degree = tf.reduce_sum(reshape_adj,1)
            reshape_adj_degree, _ = tf.nn.top_k(reshape_adj_degree, k=self.desired_size)
            # 3. store them into a list~
            origin_adjs_degree.append(reshape_adj_degree)

        ## step.2 对所有 degree distribution 取平均~~ 以保证每一个degree distribution 落在相应范围内
        self.all_layer_degree = tf.add_n(inputs=origin_adjs_degree)/len(origin_adj_list)

        # ===================================================
        # 4. 定义 loss func, learning_rate, optimizor
        # ===================================================

        ## step.1 loss func
        """方法1: 采用 L1 Norm"""
        #self.loss = tf.reduce_mean(tf.abs(self.all_layer_degree - self.logits))
        """方法2: 采用 degree distribution 之间的 KL 距离"""
        self.logits_degree = self.logits_degree + 0.000001 # 保证分母不为 0
        y = self.all_layer_degree/self.logits_degree
        self.loss = tf.abs(tf.reduce_mean(-tf.nn.softmax_cross_entropy_with_logits(logits=self.logits_degree, labels=y)))

        ## step.2 learning rate
        """learning rate decay... from TF_API"""
        start_learning_rate = 0.1
        global_step = tf.Variable(0, trainable=False)
        decay_step = 1000
        decay_rate = 0.96
        learning_rate = tf.train.exponential_decay(learning_rate=start_learning_rate,
                                                    global_step=global_step,
                                                    decay_steps=decay_step,
                                                    decay_rate=decay_rate)

        ## step.3 optimizer
        #self.opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)
        self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

    def train(self, training_step=500):
        """
        @input training_step 需要训练的次数
        @return weight & reconstructed adj
        """
        tf.global_variables_initializer().run()

        for step in range(training_step):
            self.sess.run(self.opt)
            info = ''
            for idx in range(len(self.layer_weight_list)):
                tmp = 'W_%d: [%.4f] | '%(idx, self.sess.run(self.layer_weight_list[idx]))
                info += tmp
            info += " bias: [%.4f]"%self.sess.run(self.bias)
            print('step: [%d]/[%d], loss value: %.4f'%(step+1, training_step, self.sess.run(self.loss)), info)

            if self.sess.run(self.loss) <= 0.01:
                break

        weight_list = [self.sess.run(self.layer_weight_list[idx]) for idx in range(len(self.layer_weight_list))]
        tmp_re_adj_raw = self.sess.run(self.logits)
        maxValue = tf.reduce_max(tmp_re_adj_raw)
        tmp_re_adj_raw_norm = tmp_re_adj_raw/maxValue
        tmp_re_adj_raw = tmp_re_adj_raw_norm - 0.5*tf.ones(shape = tmp_re_adj_raw.shape)

        reconstructed_Adj = tf.sigmoid(tmp_re_adj_raw)
        return weight_list, reconstructed_Adj.eval()

    def __reshapeAdj(self, oriAdj, desired_size):
        """
        @purpose: 投影到 desired_size 上
        @input: oriAdj --- 原始矩阵 / desired_size --- 期望矩阵大小
        @output: reshapeAdj
        """
        if debugFlag is True:
            print('*** running reshape...')
            start = time.time()

        # step.1 flatten matrix~
        linear_current_adj = tf.reshape(oriAdj,[-1])
        linear_length = linear_current_adj.shape.as_list()[0]

        # step.1 获取原始矩阵的size
        origin_size = oriAdj.shape.as_list()[0]
        # step.2 按片映射矩阵
        """Y=AX"""
        linear_reshapeAdj = tf.constant([],dtype=tf.float32)
        for slice in range(self.desired_size**2):
            slice_TransformM = tf.truncated_normal(shape=[linear_length],mean=0.0,stddev=1.0)
            # adj_current_column = oriAdj[:,slice]
            # value = tf.reduce_sum(slice_TransformM * adj_current_column)
            value = tf.reduce_sum(slice_TransformM * linear_current_adj, keep_dims=True)
            linear_reshapeAdj = tf.concat([linear_reshapeAdj, value],axis=0)

        reshape_adj = tf.reshape(linear_reshapeAdj, shape=[self.desired_size, self.desired_size])

        if debugFlag is True:
            print('reshaped adj size: ', reshape_adj.shape, end='\t')
            print('used time: ', time.time()-start)


        return reshape_adj


