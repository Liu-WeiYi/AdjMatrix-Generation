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
    @purpose: Given any topology, generate one with similar topology
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
                link_possibility=0.5,
                trainedFlag=False
            ):
        """
        @purpose
            set all hyperparameters
        @inputs:
            sess:               Current Tensorflow Session
            dataset_name:       Current Dataset Name
            epoch:              Epochs Number for Whole Datsets [20]
            learning_rate:      Init Learning Rate for Adam [0.0002]
            Momentum:           Momentum Value for ADAM Opt~ [0.5] --- based on DCGAN
            batch_size:         How many adj we take in for one Calculation [10]
            generatorFilter:    GENERATOR Filter Number [50]
            discriminatorFilter:DISCRIMINATOR Filter Number [50]
            generatorFC:        GENERATOR Fully Connected Layer Neural Number [1024]
            discriminatorFC:    DISCRIMINATOR Fully Connected Layer Neural Number [1024]
            training_info_dir:  [trainable_data_size, inputMatSize] Location
            OutDegree_Length:   Out Degree Length, Used for CONDITION [28]  --- Remain for Condition Graph Generator
            inputPartitionDIR:  Results DIR [facebook] --- Suggested to be same with dataset_name
            checkpointDIR:      Checkpoint Location [checkpoint]
            sampleDIR:          Sampled Network Output DIR [samples] --- Check the middle status, No use At Last :)
            reconstructDIR:     Reconstructed Layers DIR [reconstruction]
            trainedFlag:        If we need to train each layer.At first we do not have trained data~ [False]
        """
        # GAN-based Initial...
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
        # input / output size Initial...
        train_list = pickle.load(open(training_info_dir,'rb'))
        self.trainable_data_size_list = [info[0] for info in train_list]
        self.inputMat_H_list          = [info[1] for info in train_list]
        self.inputMat_W_list          = [info[1] for info in train_list]
        self.outputMat_H_list         = [info[1] for info in train_list]
        self.outputMat_W_list         = [info[1] for info in train_list]
        self.OutDegree_Length         = OutDegree_Length
        # Generator Input Length Initial...
        self.InitSampleLength_list    = [info[1] for info in train_list]
        # paths Initial...
        self.inputPartitionDIR   = inputPartitionDIR
        self.checkpointDIR       = checkpointDIR
        self.sampleDIR           = sampleDIR
        self.reconstructDIR      = reconstructDIR
        # Re Constructed graph Link Possibility Initial...
        self.link_possibility    = link_possibility

        # Constructing each layer~
        if trainedFlag is False:
            self.per_layer_modelConstrunction()
            show_all_variables()
            print('Trained Layers Process DOWN...')

    def per_layer_modelConstrunction(self):
        print('\n============================================================================')
        print('Model Construction ...')
        print('============================================================================')

        self.reconstructNet_per_layer = []
        for layer_idx in range(len(self.trainable_data_size_list)):
            """Training each layer~"""
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
        # step.0 create reconstructed adjs
        if not os.path.exists('%s_adjs.pickle'%self.dataset_name):
            # 1. create trained_graph_list for each constructed layer
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

            # 2. create origin_adj (origin_graph_list) for each original adj-mat
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

        trained_graph_adj_list.append(origin_adj)

        # ==============================================
        # Permute Adj to generate more adjs --- The improvement is small
        # ==============================================
        # permute_number = 5
        # if debugFlag is True:
        #     print("permuting adjs... ")
        # trained_graph_adj_list = permute_adjs(trained_graph_adj_list,permute_number)
        # if debugFlag is True:
        #     print('permuted adjs number: ', len(trained_graph_adj_list))


        # step.1 create Weight for each adj~
        self.trained_graph_weight_list = []
        self.layer_weight_list = [] # for Hierarchy GAN~ 😀
        count = 0
        for adj in trained_graph_adj_list:

            layer_weight = tf.Variable(tf.random_uniform([1],minval=0,maxval=1),name="weight_%d"%count)
            self.layer_weight_list.append(layer_weight)
            adj_layer_weight = layer_weight*tf.ones(shape=adj.shape)
            self.trained_graph_weight_list.append(adj_layer_weight)

            count += 1

        # create bias
        self.bias = tf.Variable(tf.random_uniform([1],minval=0,maxval=1),name="bias")

        # step.2 logit
        tmp = [self.trained_graph_weight_list[idx]*trained_graph_adj_list[idx] for idx in range(len(self.trained_graph_weight_list))]
        self.logits = tf.add(tf.add_n(tmp,name="Layered_results"),self.bias)

        # step.3 loss
        origin_adj = tf.to_float(origin_adj)
        """KL~"""
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
        @input training_step training numbers
        @return weight_list，reconstructed adj
        """
        tf.global_variables_initializer().run()

        for step in range(training_step):
            self.sess.run(self.opt)
            info = ''
            for idx in range(len(self.layer_weight_list)):
                tmp = 'W_%d: [%.4f] | '%(idx, self.sess.run(self.layer_weight_list[idx]))
                info += tmp
            info += " bias: [%.4f]"%self.sess.run(self.bias)
            # info = ['W_%d: %.4f'%(idx,self.sess.run(self.layer_weight_list[idx]) for idx in range(len(self.layer_weight_list))]
            # print('step: [%d]/[%d], loss value: %.4f'%(step+1, training_step, self.sess.run(self.loss)), info)
            print('step: [%d]/[%d], loss value: %.4f'%(step+1, training_step, self.sess.run(self.loss)))

            if self.sess.run(self.loss) <= 0.01:
                break

        weight_list = [self.sess.run(self.layer_weight_list[idx]) for idx in range(len(self.layer_weight_list))]
        # reconstructed_Adj = tf.nn.softmax(self.sess.run(self.logits))
        tmp_re_adj_raw = self.sess.run(self.logits)
        # meanValue = tf.reduce_mean(tmp_re_adj_raw)
        # tmp_re_adj_raw = tmp_re_adj_raw - meanValue*tf.ones(shape = tmp_re_adj_raw.shape)
        maxValue = tf.reduce_max(tmp_re_adj_raw)
        tmp_re_adj_raw_norm = tmp_re_adj_raw/maxValue
        tmp_re_adj_raw = tmp_re_adj_raw_norm - 0.5*tf.ones(shape = tmp_re_adj_raw.shape)

        reconstructed_Adj = tf.sigmoid(tmp_re_adj_raw)
        return weight_list, reconstructed_Adj.eval()


