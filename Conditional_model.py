#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Condition model (based on DCGAN)
  Created:  04/08/17
"""
import numpy as np
import tensorflow as tf
import time
import os
import sys
import pickle
import glob
import math
import networkx as nx

from utils import *

debugFlag = False

class Condition_adjMatrix_Generator(object):
    """
    @purpose: 随便给定一个网络，生成具有对应特性的网络
    """
    def __init__(self,
                sess, dataset_name,
                epoch=10, learning_rate=0.0002, Momentum=0.5,
                batch_size=10,
                generatorFilter=50, discriminatorFilter=50,
                generatorFC=1024, discriminatorFC=1024,
                trainable_data_size=20000,
                inputMat_H=28, inputMat_W=28, outputMat_H=28, outputMat_W=28,
                OutDegree_Length=28, InitGen_Length=28,
                inputPartitionDIR="WS_test", checkpointDIR="checkpoint", sampleDIR="samples", reconstructDIR="reconstruction",
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
            Momentum:           Momentum Value for ADAM Opt~ [0.5] --- based on DCGAN
            batch_size:         How many adj we take in for one Calculation [10]
            generatorFilter:    GENERATOR Filter Number [50]
            discriminatorFilter:DISCRIMINATOR Filter Number [50]
            generatorFC:        GENERATOR Fully Connected Layer Neural Number [1024]
            discriminatorFC:    DISCRIMINATOR Fully Connected Layer Neural Number [1024]
            trainable_data_size:Trainable data size [20000] --- NOT USE NOW!!
            inputMat_H:         Input Adj Height [28] --- p.s. Suggested to be divided by 4 as stride=2, and we use TWO layer (de-)conv layers
            inputMat_W:         Input Adj Width [28] --- p.s. the same ↑↑↑
            outputMat_H:        Output Adj Height [28] --- p.s. the same ↑↑↑
            outputMat_W:        Output Adj Width [28] --- p.s. the same ↑↑↑
            OutDegree_Length:   Out Degree Length, Used for CONDITION [28]  --- Remain for Condition Graph Generator
            InitGen_Length:     GENERATOR Input, Sampled from [0,1] uniform Distribution
            inputPartitionDIR:  Results DIR [facebook] --- Suggested to be same with dataset_name
            checkpointDIR:      Checkpoint Location [checkpoint]
            sampleDIR:          Sampled Network Output DIR [samples] --- Check the middle status, No use At Last :)
            reconstructDIR:     Reconstructed Layers DIR [reconstruction]
            link_possibility:   Create an edge in reconstructed ADJ ONLY IF Current Value ≥ link_possibility
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
        self.trainable_data_size = trainable_data_size
        self.inputMat_H          = inputMat_H
        self.inputMat_W          = inputMat_W
        self.outputMat_H         = outputMat_H
        self.outputMat_W         = outputMat_W
        self.OutDegreeLength     = OutDegree_Length
        # Generator Input Length Initial...
        self.InitSampleLength    = InitGen_Length
        # paths Initial...
        self.inputPartitionDIR   = inputPartitionDIR
        self.checkpointDIR       = checkpointDIR
        self.sampleDIR           = sampleDIR
        self.reconstructDIR      = reconstructDIR
        # Re Constructed graph Link Possibility Initial...
        self.link_possibility    = link_possibility

        """
        pre-define batch norm --- based on DCGAN
        """
        self.generator_batch_norm_0 = batch_norm(name="g_bn0_%d"%self.inputMat_H)
        self.generator_batch_norm_1 = batch_norm(name="g_bn1_%d"%self.inputMat_H)
        self.generator_batch_norm_2 = batch_norm(name="g_bn2_%d"%self.inputMat_H)

        self.discriminator_batch_norm1 = batch_norm(name="d_bn1_%d"%self.inputMat_H)
        self.discriminator_batch_norm2 = batch_norm(name="d_bn2_%d"%self.inputMat_H)

        # Create GAN~
        self.modelConstrunction()

    def modelConstrunction(self):
        print('\n============================================================================')
        print('Model Construction ...')
        print('============================================================================')
        # 1. place holder
        ## for Real Data
        self.OutDegreeVector = tf.placeholder(tf.float32, shape=[self.batch_size, self.OutDegreeLength], name="Out_Degree_Vector_%d"%self.inputMat_H)
        self.inputMat = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputMat_H, self.inputMat_W, 1], name="Real_Input_Adj_Matrix_%d"%self.inputMat_H)

        ## for Generator INPUT
        self.InitSampleVector = tf.placeholder(tf.float32, shape=[None, self.InitSampleLength], name="GEN_Input_%d"%self.inputMat_H)

        # 2. Generator
        self.Generator = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector, trainFlag=True)

        # 3. Sampler --- use for sample network in middle process~ :) No use at last :)
        self.Sampler = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector, trainFlag=False)

        # 4. Reconstruction --- use for reconstructing adj
        self.re_Mat = tf.placeholder(tf.float32, shape=[1, self.inputMat_H, self.inputMat_W, 1], name="Reconstruct_Input_Adj_Matrix_%d"%self.inputMat_H)
        self.re_OutDegreeVector = tf.placeholder(tf.float32, shape=[1, self.OutDegreeLength], name="Reconstruct_Out_Degree_Vector_%d"%self.inputMat_H)
        self.re_Construction = self.__re_Construction(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector= self.re_OutDegreeVector, trainFlag=False)

        # 4. Discriminator_real
        self.Discriminator_real, self.D_logits_real = self.__discriminator(
            InputAdjMatrix=self.inputMat,
            Input_OutDegreeVector = self.OutDegreeVector,
            reuse_variables = False
        )
        # 5. Discriminator_fake
        self.Discriminator_fake, self.D_logits_fake = self.__discriminator(
            InputAdjMatrix=self.Generator,
            Input_OutDegreeVector=self.OutDegreeVector,
            reuse_variables=True
        )

        # 6. Discriminator LOSS FUNCTION
        ## Learn real datasets --- ∴ using tf.ones()
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real,labels=tf.ones_like(self.Discriminator_real))
        )
        ## Learn fake datasets --- ∴ using tf.zeros()
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,labels=tf.zeros_like(self.Discriminator_fake))
        )
        self.discriminator_loss = self.d_loss_real + self.d_loss_fake

        # 7. Generator LOSS FUNCTION
        # March Towards to REAL Data  --- ∴ using tf.ones()
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Discriminator_fake,labels=tf.ones_like(self.D_logits_fake))
        )

        # --> TensorBoard
        self.InitSampleVector_sum = tf.summary.histogram(name="GEN Input_%d"%self.inputMat_H,values=self.InitSampleVector)
        self.Generator_sum = tf.summary.histogram(name="GEN Output_%d"%self.inputMat_H,values=self.Generator)
        self.Discriminator_real_sum = tf.summary.histogram(name="Discriminator Real Output_%d"%self.inputMat_H, values=self.Discriminator_real)
        self.Discriminator_fake_sum = tf.summary.histogram(name="Discriminator Fake Output_%d"%self.inputMat_H, values=self.Discriminator_fake)

        self.d_loss_real_sum = tf.summary.scalar(name="Discriminator Real Loss_%d"%self.inputMat_H, tensor=self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar(name="Discriminator Fake Loss_%d"%self.inputMat_H, tensor=self.d_loss_fake)
        self.discriminator_loss_sum = tf.summary.scalar(name="Discriminator Loss_%d"%self.inputMat_H, tensor=self.discriminator_loss)

        self.generator_loss_sum = tf.summary.scalar(name="Generator Loss_%d"%self.inputMat_H,tensor=self.generator_loss)

        # --> split generator vars & discriminator vars --- needed step! useful in trainning~~
        t_vars = tf.trainable_variables()

        self.discriminator_vars = [var for var in t_vars if "d_" in var.name]
        self.generator_vars = [var for var in t_vars if "g_" in var.name]

        # save checkpoint...
        self.saver = tf.train.Saver()

        print('down ...')

    def train(self,returnFlag=False):
        # IF returnFlag=True, Then RETURN reconstructed graph
        print('\n============================================================================')
        print('Train Begin...')
        print('============================================================================')
        # 1. define discriminator & generator opt
        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.Momentum)\
            .minimize(self.discriminator_loss, var_list=self.discriminator_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.Momentum)\
            .minimize(self.generator_loss, var_list = self.generator_vars)

        # 2. init all variables~ --- tf
        tf.global_variables_initializer().run()

        # 3. record all variables~
        self.generator_related_sum = tf.summary.merge([ self.InitSampleVector_sum,
                                                        self.Discriminator_fake_sum,
                                                        self.Generator_sum,
                                                        self.d_loss_fake_sum,
                                                        self.generator_loss_sum
                                                        ])
        self.discriminator_related_sum = tf.summary.merge([ self.InitSampleVector_sum,
                                                            self.Discriminator_real_sum,
                                                            self.d_loss_real_sum,
                                                            self.discriminator_loss_sum
                                                            ])
        self.writer = tf.summary.FileWriter("./boards", self.sess.graph)

        # 4. Input Real adj-mat & Outdegree --- Dispose...
        """
        Consider Mem size
        Instead of reading them once for all, We read data seperately :)
        """
        # data_mat, data_degree = self.__load_AdjMatInfo(AdjMat_OutDegree_Dir = self.inputPartitionDIR,startPoint=0,endPoint=20000,MatSize=self.inputMat_H)

        # 5. checkpoint
        could_load, checkpoint_counter = self.load(self.checkpointDIR)
        if could_load is True:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 6. Ready... LET'S FLY!!
        counter = 0
        start_time = time.time()
        for epoch in range(self.epoch):
            # step = data_mat.shape[0] // self.batch_size
            step  = self.trainable_data_size // self.batch_size

            for idx in range(0,step):
                # batch_data_mat          = data_mat[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch_data_degree       = data_degree[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_data_mat, batch_data_degree = self.__load_AdjMatInfo( AdjMat_OutDegree_Dir = self.inputPartitionDIR,
                                                                            startPoint=idx*self.batch_size,
                                                                            endPoint=(idx+1)*self.batch_size,
                                                                            MatSize=self.inputMat_H)
                ## 1. Create Generator Input
                batch_generator_input   = np.random.uniform(low=-1,high=1,size=[self.batch_size, self.InitSampleLength]).astype(np.float32)

                ## 2. Update Discriminator
                _, summary_str = self.sess.run( [d_optimizer, self.discriminator_related_sum],
                                                feed_dict={
                                                    self.inputMat: batch_data_mat,
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 3. Update Generator
                _, summary_str = self.sess.run( [g_optimizer, self.generator_related_sum],
                                                feed_dict={
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 4. Run g_optimizer twice to make sure that d_loss does not go to zero -- based on DCGAN
                _, summary_str = self.sess.run( [g_optimizer, self.generator_related_sum],
                                                feed_dict={
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 5. Record loss for Each Time --- based on DCGAN
                errD_fake = self.d_loss_fake.eval({
                    self.InitSampleVector : batch_generator_input,
                    self.OutDegreeVector : batch_data_degree
                })
                errD_real = self.d_loss_real.eval({
                    self.inputMat : batch_data_mat,
                    self.OutDegreeVector : batch_data_degree
                })
                errG = self.generator_loss.eval({
                    self.InitSampleVector : batch_generator_input,
                    self.OutDegreeVector : batch_data_degree
                })

                ## 6. Output to Console
                ### 1. Loss Value
                counter += 1
                print("Epoch: [%d] | [%d/%d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"%(epoch, idx+1, step, time.time()-start_time, errD_fake+errD_real, errG))
                ### 2. Create a Sample every 500 times~~
                """BEGIN: --- No use at last!! Only to check middle statue !!"""
                if counter % 500 == 0:
                    ## Generator Input
                    sample_input = np.random.uniform(low=-1, high=1, size=[self.batch_size, self.InitSampleLength])
                    if debugFlag is True:
                        print('sampled input for Generator shape: ', sample_input.shape) # (20, 28)
                    # sample_mat = data_mat[0:self.batch_size]
                    # sample_labels = data_degree[0:self.batch_size]
                    sample_mat, sample_labels = self.__load_AdjMatInfo( AdjMat_OutDegree_Dir = self.inputPartitionDIR,
                                                                        startPoint=0,
                                                                        endPoint=self.batch_size,
                                                                        MatSize=self.inputMat_H)

                    samples, d_loss, g_loss = self.sess.run([self.Sampler, self.discriminator_loss, self.generator_loss],
                                                            feed_dict={
                                                                self.InitSampleVector : sample_input,
                                                                self.inputMat : sample_mat,
                                                                self.OutDegreeVector : sample_labels
                                                            })

                    sample_folder = os.path.join(self.sampleDIR, "%s_%d_%d_%.1f"%(self.dataset_name, self.inputMat_H,self.trainable_data_size, self.link_possibility))
                    sample_folder = os.path.join(sample_folder,'')
                    """example: sample_folder = samples/WS_test_28_20000_0.5"""
                    save_topology(  adj=samples,
                                    path=sample_folder, graph_name = 'train_%d_%d'%(epoch,counter),
                                    link_possibility = self.link_possibility)
                    print("[Sample] d_loss: %.8f, g_loss: %.8f"%(d_loss, g_loss))
                """END: --- No use at last!! Only to check middle statue !!"""

                if counter % 500 == 0:
                    # checkpointDIR = self.checkpointDIR
                    self.save(self.checkpointDIR,counter)

    def saveModel(self):
        """
        Save Trained Model
        """
        save_path = os.path.join('trained_Model', "%s_%d_%d_%.1f"%(self.dataset_name, self.inputMat_H,self.trainable_data_size, self.link_possibility),'')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        saver = tf.train.Saver()
        saver.save(self.sess, save_path)
        return save_path

    def reconstructMat(self, type="Hierarchy"):
        print('\n============================================================================')
        print('Re-Construction Graph...')
        print('============================================================================')

        # step.0 reconstruct every subgraph...
        each_part = {}
        for i in range(self.trainable_data_size):
            # for generator input~
            """as we generate the subgraph one by one, then
            please attension that in here the batch size is 1 !!!"""
            sample_input = np.random.uniform(low=-1, high=1.0, size=[1, self.InitSampleLength])

            partition_mat,partition_labels = self.__load_AdjMatInfo(AdjMat_OutDegree_Dir = self.inputPartitionDIR,
                                                                    startPoint=i,
                                                                    endPoint=i+1,
                                                                    MatSize=self.inputMat_H)

            reconstruct_adj = self.sess.run([self.re_Construction],
                                            feed_dict={
                                                self.InitSampleVector : sample_input,
                                                self.re_Mat : partition_mat,
                                                self.re_OutDegreeVector : partition_labels
                                            })

            each_part[i] = np.squeeze(reconstruct_adj[0])

        # step.1  use reconstructed reconstruct_adj and
        # mapping file <filename>_MatSize.map (part2Node) to attach all parts
        if type == "Hierarchy":
            reconstruct_folder = os.path.join(self.reconstructDIR, self.dataset_name, "Hierarchy",'')
            if not os.path.exists(reconstruct_folder):
                os.makedirs(reconstruct_folder)

            graph_name ="%s_%d_%d"%(self.dataset_name, self.inputMat_H,self.trainable_data_size)
            path = os.path.join('data',self.dataset_name,'')
            Node2Part = pickle.load(open(glob.glob(path+"*_%d.map"%self.inputMat_H)[0],'rb'))

            weighted_graph = construct_topology(graph_name, each_part, Node2Part)
            pickle.dump(weighted_graph, open(reconstruct_folder+graph_name+".nxgraph", 'wb'))


        elif type == "Condition":
            # under construction ... 😶
            pass

    def __generator(self, InitInputSampleVector, Input_OutDegreeVector, trainFlag=True):
        """
        @porpose Generator~~
        @Arch: RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu
                -> generatorFC -> batch_norm_1 -> relu
                -> batch_norm -> deconv -> batch_norm_2 -> relu
                -> deconv -> batch_norm3 -> sigmoid -> generated_Adj-Matrix

        @InitInputSampleVector : random Vector for Gen
        @Input_OutDegreeVector : OutputDegreeVector ---Used for Condition GAN
        @trainFlag             : (When we sample the graph, there is no need for training!)
        """
        with tf.variable_scope("generator") as scope:
            if trainFlag is False:
                scope.reuse_variables()

            # 1. get Output Subgraph Width/Height
            Height, Width = self.outputMat_H, self.outputMat_W

            deconv1_Height, deconv1_Width = int(math.ceil(Height/2)), int(math.ceil(Width/2))
            deconv2_Height, deconv2_Weight = int(math.ceil(deconv1_Height/2)), int(math.ceil(deconv1_Width/2))

            #  2.  RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu := h0 -> h0+OutDegreeVector
            ## 2.1 Consider CONDITION~
            input = tf.concat([InitInputSampleVector, Input_OutDegreeVector], axis=1)
            if debugFlag is True:
                if trainFlag is True:
                    print('============================= generator =============================')
                elif trainFlag is False:
                    print('============================= sampler =============================')
                print('input shape: ',input.shape) # (20, 56)
            ## 2.2 FC+BatchNorm+relu
            h0=tf.nn.relu(
                self.generator_batch_norm_0(
                    linear(input_=input, output_size=self.generatorFC, scope="g_h0_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
                )
            if debugFlag is True:
                print('h0 shape: ', h0.shape) # (20, 1024)
            ## 2.3 Consider CONDITION~
            h0 = tf.concat([h0, Input_OutDegreeVector], axis=1)
            if debugFlag is True:
                print('h0+outdegree shape: ', h0.shape) # (20, 1052)

            # 3. h0+OutDegreeVector -> generatorFC -> batch_norm_1 -> relu := h1 -> reshape(h1) -> h1+OutDegreeVector
            output_size = self.generatorFilter*2 * deconv2_Height * deconv2_Weight
            h1 = tf.nn.relu(
                self.generator_batch_norm_1(
                    linear(input_=h0, output_size=output_size, scope="g_h1_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('h1 shape: ', h1.shape) # (20, 4900)
            #  reshape vector-h1 to tensor-h1
            h1 = tf.reshape(h1, [self.batch_size, deconv2_Height, deconv2_Weight, self.generatorFilter*2])
            if debugFlag is True:
                print('reshape h1: ', h1.shape) # (20, 7,7, 100)

            OutDegreeTensor = tf.reshape(self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('reshape OutDegreeVector: ', OutDegreeTensor.shape) # (20, 1, 1, 28)
            # Consider CONDITION~
            h1 = conv_cond_concat(x=h1, y=OutDegreeTensor)
            if debugFlag is True:
                print('reshape h1 + OutDegreeVector: ', h1.shape) # (20, 7, 7, 100+28)

            # 4. h1+OutDegreeVector -> deconv2d -> batch_norm_2 -> relu := h2 -> h2+OutDegreeVector
            h2 = tf.nn.relu(
                self.generator_batch_norm_2(
                    deconv2d(input_=h1, output_shape=[self.batch_size, deconv1_Height, deconv1_Width, self.generatorFilter*2], name="g_h2_%d"%self.inputMat_H),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('deconv h2: ', h2.shape) # (20, 14, 14, 100)
            # Consider CONDITION~
            h2 = conv_cond_concat(x=h2, y=OutDegreeTensor)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (20, 14, 14, 128)

            # 5. h2+OutDegreeVector -> deconv2d -> sigmoid
            """There is no need to batch norm for the last layer!! As we want to preserve all the information"""
            h3 = tf.nn.sigmoid(
                deconv2d(input_=h2, output_shape=[self.batch_size, Height, Width, 1], name="g_h3_%d"%self.inputMat_H)
            )
            if debugFlag is True:
                print('final layer h3 shape: ', h3.shape) # (20, 28, 28, 1)

            return h3

    def __discriminator(self, InputAdjMatrix, Input_OutDegreeVector, reuse_variables=False):
        """
        @porpose Discriminator
        @Disc Arch: InputAdjMatrix+outDegreeVector -> condition(OutDegreeVector) & conv2d -> leaky_relu
                        -> condition(OutDegreeVector) & conv2d -> batch_norm_1 -> leaky_relu -> linear
                            -> FC -> batch_norm_2 -> leaky_relu
                                -> map_to_One_value -> sigmoid

        @InputAdjMatrix         : adj-mat for input
        @Input_OutDegreeVector  : OutputDegreeVector ---Used for Condition GAN
        @reuse_variables        : As there will be D_real and D_fake. So we need to reuse the vars
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse_variables is True:
                scope.reuse_variables()

            if debugFlag is True:
                print('============================= discriminator =============================')
                print('input shape: ',InputAdjMatrix.shape) # (20, 28, 28, 1)
            # 1. Consider CONDITION~
            OutDegreeTensor = tf.reshape(tensor=self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('outdegree tensor: ', OutDegreeTensor.shape) # (20, 1,1, 28)
            input_ = conv_cond_concat(x=InputAdjMatrix, y=OutDegreeTensor) # (20, 28,28, 1+28)
            if debugFlag is True:
                print('input+outdegree vector shape: ',input_.shape) # (20, 28,28, 1+28)

            # 2. First Hidden_Layer: InputAdjMatrix+outDegreeVector -> condition(OutDegreeVector) & conv2d -> leaky_relu := h0 -> h0+outDegreeVector
            h0 = lrelu(conv2d(input_=input_, output_filter=1+self.OutDegreeLength, name="d_h0_conv_%d"%self.inputMat_H))
            if debugFlag is True:
                print('CONDITION h0 shape: ', h0.shape) # (20, 14,14, 29)
            h0 = conv_cond_concat(x=h0,y=OutDegreeTensor)
            if debugFlag is True:
                print('h0+OutDegreeVector: ', h0.shape) # (20, 14,14, 29+28=57)

            # 3. Second Hidden_Layer: h0+outDegreeVector -> condition(OutDegreeVector) & conv2d -> batch_norm_1 -> leaky_relu -> linear := h1
            h1 = lrelu(self.discriminator_batch_norm1(
                conv2d(input_=h0, output_filter = self.discriminatorFilter + self.OutDegreeLength, name="d_h1_conv1_%d"%self.inputMat_H)
                                            )
                        )
            if debugFlag is True:
                print('CONDITION h1 shape: ', h1.shape) # (20, 7,7, 50+28)
            ## Prepare Linear Tensor for FC layer
            h1 = tf.reshape(tensor=h1, shape=[self.batch_size, -1])
            if debugFlag is True:
                print('linear h1: ', h1.shape) # (20, 7*7*78=3822)

            h1 = tf.concat(values=[h1,self.OutDegreeVector],axis=1)
            if debugFlag is True:
                print('h1+OutDegreeVector: ', h1.shape) # (20, 3822+28=3850)

            # 4. First FC layer: h1 -> FC -> batch_norm_2 -> leaky_relu := h2
            h2 = lrelu(self.discriminator_batch_norm2(linear(input_=h1, output_size=self.discriminatorFC, scope="d_h2_lin_%d"%self.inputMat_H)))
            if debugFlag is True:
                print('FC h2 shape: ', h2.shape) # (20, 1024)
            ## Consider CONDITION~
            h2 = tf.concat(values=[h2,self.OutDegreeVector], axis=1) # (20, 1024+28=1052)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (20, 1052)

            # 5. h3 -> map_to_One_value -> sigmoid
            h3 = linear(input_=h2, output_size=1, scope="d_h3_lin_%d"%self.inputMat_H)
            if debugFlag is True:
                print('h3 map to ONE Value: ', h3.shape) # (20,1)
            h3_sigmoid = tf.nn.sigmoid(h3,name="sigmoid_h3_%d"%self.inputMat_H)
            if debugFlag is True:
                print('h3 to sigmoid: ', h3_sigmoid.shape) # (20, 1)

            return h3_sigmoid, h3

    def __re_Construction(self, InitInputSampleVector, Input_OutDegreeVector, trainFlag=False):
        """
        @porpose reconstructor
        It has the same structure with Generator. We just copy all the code from Generator :)
        """
        with tf.variable_scope("generator") as scope:
            if trainFlag is False:
                scope.reuse_variables()

            Height, Width = self.outputMat_H, self.outputMat_W
            """p.s. 为保证/4能够除尽，这里引入 math.ceil对结果进行向上取整"""
            deconv1_Height, deconv1_Width = int(math.ceil(Height/2)), int(math.ceil(Width/2))
            deconv2_Height, deconv2_Weight = int(math.ceil(deconv1_Height/2)), int(math.ceil(deconv1_Width/2))

            #  2.  RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu := h0 -> h0+OutDegreeVector
            """p.s. 下面所有的所谓"向量"其实都是一个矩阵，其第一个维度就是batch_size"""

            input = tf.concat([InitInputSampleVector, Input_OutDegreeVector], axis=1) # <- input 为一个 28+28 共 56 的 向量
            if debugFlag is True:
                if trainFlag is True:
                    print('============================= generator =============================')
                elif trainFlag is False:
                    print('============================= ReConstruction =============================')
                print('input shape: ',input.shape) # (20, 56)

            h0=tf.nn.relu(
                self.generator_batch_norm_0(
                    linear(input_=input, output_size=self.generatorFC, scope="g_h0_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
                )
            if debugFlag is True:
                print('h0 shape: ', h0.shape) # (20, 1024)

            h0 = tf.concat([h0, Input_OutDegreeVector], axis=1)
            if debugFlag is True:
                print('h0+outdegree shape: ', h0.shape) # (20, 1052)

            # 3. h0+OutDegreeVector -> generatorFC -> batch_norm_1 -> relu := h1 -> reshape(h1) -> h1+OutDegreeVector
            output_size = self.generatorFilter*2 * deconv2_Height * deconv2_Weight
            h1 = tf.nn.relu(
                self.generator_batch_norm_1(
                    linear(input_=h0, output_size=output_size, scope="g_h1_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('h1 shape: ', h1.shape) # (20, 4900)

            h1 = tf.reshape(h1, [1, deconv2_Height, deconv2_Weight, self.generatorFilter*2])
            if debugFlag is True:
                print('reshape h1: ', h1.shape) # (20, 7,7, 100)

            OutDegreeTensor = tf.reshape(self.re_OutDegreeVector, shape=[1, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('reshape OutDegreeVector: ', OutDegreeTensor.shape) # (20, 1, 1, 28)

            h1 = conv_cond_concat(x=h1, y=OutDegreeTensor)
            if debugFlag is True:
                print('reshape h1 + OutDegreeVector: ', h1.shape) # (20, 7, 7, 100+28)

            # 4. h1+OutDegreeVector -> deconv2d -> batch_norm_2 -> relu := h2 -> h2+OutDegreeVector
            h2 = tf.nn.relu(
                self.generator_batch_norm_2(
                    deconv2d(input_=h1, output_shape=[1, deconv1_Height, deconv1_Width, self.generatorFilter*2], name="g_h2_%d"%self.inputMat_H),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('deconv h2: ', h2.shape) # (20, 14, 14, 100)

            h2 = conv_cond_concat(x=h2, y=OutDegreeTensor)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (20, 14, 14, 128)

            # 5. h2+OutDegreeVector -> deconv2d -> sigmoid # WITH NO batch_norm for the last layer!!!
            h3 = tf.nn.sigmoid(
                deconv2d(input_=h2, output_shape=[1, Height, Width, 1], name="g_h3_%d"%self.inputMat_H)
            )
            if debugFlag is True:
                print('final layer h3 shape: ', h3.shape) # (20, 28, 28, 1)

            return h3

    def __load_AdjMatInfo(self, AdjMat_OutDegree_Dir, startPoint=0, endPoint=50, MatSize=-1):
        if MatSize == -1:
            sys.exit('[!!!] Please specify mat size... otherwise the program cannot understand which partition results is needed.')
        path = os.path.join('data',AdjMat_OutDegree_Dir)
        path = os.path.join(path, '')
        if debugFlag is True:
            print('current data file path: ', path)
        adj_file = glob.glob(path+'*_%d.graph'%MatSize)[0]
        if debugFlag is True:
            print('current adj file dir: ', adj_file, end='\t')
        degree_file = glob.glob(path+'*_%d.degree'%MatSize)[0]

        # read adj-dict
        adj = pickle.load(open(adj_file,'rb'))
        degree = pickle.load(open(degree_file,'rb'))
        if debugFlag is True:
            print('loaded adj file, adj.keys() = ', len(adj.keys()), end='\t')
            print('loaded degree file, degree.keys() = ', len(degree.keys()), end='\t')
            print('start point= ', startPoint, '\tend point= ', endPoint)
        # adj to Tensor
        for i in range(startPoint, endPoint):
            if i == startPoint:
                """ fixed bug --- i may exceed the adj.keys() """
                if i not in adj.keys():
                    i = len(adj.keys())-1
                #Tensor = tf.stack([adj[i]],axis=0)
                Tensor = np.stack([adj[i]],axis=0)
                #DegreeTensor = tf.stack([degree[i]],axis=0)
                DegreeTensor = np.stack([degree[i]],axis=0)
            else:
                """ fixed bug --- i may exceed the adj.keys() """
                if i not in adj.keys():
                    i = len(adj.keys())-1
                #slice = tf.stack([adj[i]],axis=0)
                slice = np.stack([adj[i]],axis=0)
                #slice_degree = tf.stack([degree[i]],axis=0)
                slice_degree = np.stack([degree[i]],axis=0)
                #Tensor = tf.concat([Tensor, slice],axis=0)
                Tensor = np.concatenate([Tensor, slice],axis=0)
                #DegreeTensor = tf.concat([DegreeTensor, slice_degree],axis=0)
                DegreeTensor = np.concatenate([DegreeTensor, slice_degree],axis=0)

        #AdjMatTensor = tf.expand_dims(input=Tensor, axis=-1)
        AdjMatTensor = np.expand_dims(Tensor, axis=-1)
        if debugFlag is True:
            print('Output -> AdjMat Tensor shape: ', AdjMatTensor.shape, end='\t')
            print('Output -> Degree Tensor shape: ', DegreeTensor.shape)

        #return AdjMatTensor.eval(),DegreeTensor.eval()
        return AdjMatTensor,DegreeTensor

    # ================================================
    # FROM DCGAN
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset_name, self.inputMat_H, self.trainable_data_size, self.link_possibility)

    def save(self, checkpoint_dir, step):
        # model_name = "DCGAN.model"
        model_name = "AdjMatrixGenerator.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,os.path.join(checkpoint_dir, model_name),global_step=step)

    def load(self, checkpoint_dir):
        import re
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0
    # From DCGAN END
    # ================================================



