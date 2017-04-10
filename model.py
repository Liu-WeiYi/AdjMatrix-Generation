#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  model (based on DCGAN)
  Created: 04/08/17
"""
import numpy as np
import tensorflow as tf
import time
import os
import pickle
import glob

from utils import *

debugFlag = True

class adj_Matrix_Generator(object):
    """
    主类
    """
    def __init__(self,
                 sess, dataset_name,
                 epoch=10, learning_rate=0.0002, Momentum=0.5,
                 batch_size=64,
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
            batch_size:         毝一次读入的adj-matrix数針 [64]
            generatorFilter:    生戝器初始的filter数值 [64] --- 注: 算法中毝一层的filter都设置为该值的2倝 based on DCGAN
            discriminatorFilter:判别器初始的filter数值 [64] --- 注: 算法中毝一层的filter都设置为该值的2倝 based on DCGAN
            generatorFC:        生戝器的全连接层的神绝元个数 [1024]
            discriminatorFC:    判别器的全连接层的神绝元个数 [1024]
            inputMat_H:         输入邻接矩阵的Height [100]
            inputMat_W:         输入邻接矩阵的Width [100]
            outputMat_H:        输出邻接矩阵的Height [100]
            outputMat_W:        输出邻接矩阵的Width [100]
            OutDegree_Length:   当剝AdjMatrix的出度坑針长度 [100]  --- 相当于DCGAN中的 y_lim
            InitGen_Length:     初始化GEN时的输入坑針大尝 [100] --- 相当于DCGAN中的 z_lim
            inputPartitionDIR:  分割坎的矩阵的存档点 [./graphs]
            checkpointDIR:      存档点 地址 [./checkpoint]
            sampleDIR:          输出图僝地址 [./samples]
        """
        # 坂数初始化
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
        """
        注愝，这里的inputMat 和 outputMat 均指坑的是 生戝器 坃万丝覝杞错了!
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
        因为坎面多个地方都需覝用到batch_norm擝作，因此在这里事先进行定义
        """
        self.generator_batch_norm_0 = batch_norm(name='g_bn0')
        self.generator_batch_norm_1 = batch_norm(name='g_bn1')
        self.generator_batch_norm_2 = batch_norm(name='g_bn2')

        self.discriminator_batch_norm1 = batch_norm(name="d_bn1")
        self.discriminator_batch_norm2 = batch_norm(name="d_bn2")

        # 创建GAN --- based on DCGAN
        self.modelConstrunction()

    def modelConstrunction(self):
        print('\n=================================================================================')
        print('Model Construction ...')
        # 1. place holder 装输入的坘針
        self.OutDegreeVector = tf.placeholder(tf.float32, shape=[self.batch_size, self.OutDegreeLength], name="Out_Degree_Vector")
        self.inputMat = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputMat_H, self.inputMat_W, 1], name="Real_Input_Adj_Matrix")
        self.sampleInput = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputMat_H, self.inputMat_W, 1], name="Sample_Input_Adj_Matrix")
        self.InitSampleVector = tf.placeholder(tf.float32, shape=[None, self.InitSampleLength], name="GEN_Input")

        # 2. Generator
        self.Generator = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector, trainFlag=True)

        # 3. Sampler --- 用作 生戝图片的时候~~
        self.Sampler = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector, trainFlag=False)

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
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real,labels=tf.ones_like(self.Discriminator_real))
        )
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,labels=tf.zeros_like(self.Discriminator_fake))
        )
        self.discriminator_loss = self.d_loss_real + self.d_loss_fake

        # 7. Generator LOSS FUNCTION
        """ 根杮公弝~~
        注愝 生戝器的Loss Function应该算Discriminator的LOGIT哈＝＝
        """
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Discriminator_fake,labels=tf.ones_like(self.D_logits_fake))
        )

        # --> 放入TensorBoard中
        self.InitSampleVector_sum = tf.summary.histogram(name="GEN Input",values=self.InitSampleVector)
        self.Generator_sum = tf.summary.histogram(name="GEN Output",values=self.Generator)
        self.Discriminator_real_sum = tf.summary.histogram(name="Discriminator Real Output", values=self.Discriminator_real)
        self.Discriminator_fake_sum = tf.summary.histogram(name="Discriminator Fake Output", values=self.Discriminator_fake)

        self.d_loss_real_sum = tf.summary.scalar(name="Discriminator Real Loss", tensor=self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar(name="Discriminator Fake Loss", tensor=self.d_loss_fake)
        self.discriminator_loss_sum = tf.summary.scalar(name="Discriminator Loss", tensor=self.discriminator_loss)

        self.generator_loss_sum = tf.summary.scalar(name='Generator Loss',tensor=self.generator_loss)

        # --> 存储所有坘針 --- based on DCGAN
        t_vars = tf.trainable_variables()

        self.discriminator_vars = [var for var in t_vars if "d_" in var.name]
        self.generator_vars = [var for var in t_vars if "g_" in var.name]

        self.saver = tf.train.Saver()

        print('down ...')

    def train(self):
        print('\n=================================================================================')
        print('Train Begin...')
        # 1. 定义 判别器 和 生戝器的 Loss  Function 优化方法
        d_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.Momentum)\
            .minimize(self.discriminator_loss, var_list=self.discriminator_vars)
        g_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=self.Momentum)\
            .minimize(self.generator_loss, var_list = self.generator_vars)

        # 2. init all variables~ --- tf
        tf.global_variables_initializer().run()

        # 3. record all variables~ -- based on DCGAN
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

        # 4. 准备坄秝数杮~
        ## 读入 mat 坊其 Outdegree
        """如果内存丝够~ 坯以在坎面冝读~"""
        data_mat, data_degree = self.__load_AdjMatInfo(AdjMat_OutDegree_Dir = self.inputPartitionDIR,startPoint=0,endPoint=20000)

        # 5. 记录checkpoint --- based on DCGAN
        could_load, checkpoint_counter = self.load(self.checkpointDIR)
        if could_load is True:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 6. 训练开始~
        counter = 0
        start_time = time.time() # 记录当剝开始时间~~
        for epoch in range(self.epoch):
            step = data_mat.shape[0] // self.batch_size
            for idx in range(0,step):
                ## 1. 读入所有数杮
                batch_data_mat          = data_mat[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_data_degree       = data_degree[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_generator_input   = np.random.uniform(low=-1,high=1,size=[self.batch_size, self.InitSampleLength]).astype(np.float32)

                ## 2. Update 判别器
                _, summary_str = self.sess.run( [d_optimizer, self.discriminator_related_sum],
                                                feed_dict={
                                                    self.inputMat: batch_data_mat,
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 3. Update 生戝器
                _, summary_str = self.sess.run( [g_optimizer, self.generator_related_sum],
                                                feed_dict={
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 4. Run g_optim twice to make sure that d_loss does not go to zero -- based on DCGAN
                _, summary_str = self.sess.run( [g_optimizer, self.generator_related_sum],
                                                feed_dict={
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 5. 记录毝次的loss --- based on DCGAN
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

                ## 6. 输出
                ### 1. 毝次都输出错误信杯
                counter += 1
                print("Epoch: [%d] | [%d/%d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"%(epoch, idx, step, time.time()-start_time, errD_fake+errD_real, errG))

                ### 2. 毝100次生戝一张图片。。。并保存~
                if counter % 100 == 0:

                    ## 用作Generator输入 ---  用于生戝 Mat~~ 作用坌 self.Sampler
                    sample_input = np.random.uniform(low=-1, high=1, size=[self.batch_size, self.InitSampleLength])
                    if debugFlag is True:
                        print('sampled input for Generator shape: ', sample_input.shape) # (64, 100)
                    sample_mat = data_mat[0:self.batch_size]
                    sample_labels = data_degree[0:self.batch_size]

                    samples, d_loss, g_loss = self.sess.run([self.Sampler, self.discriminator_loss, self.generator_loss],
                                                            feed_dict={
                                                                self.InitSampleVector : sample_input,
                                                                self.inputMat : sample_mat,
                                                                self.OutDegreeVector : sample_labels
                                                            })

                    save_topology(adj=samples,
                                  sample_folder=self.sampleDIR, dataset_name = self.dataset_name, graph_name = 'train_%d_%d'%(epoch,counter))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f"%(d_loss, g_loss))

                if counter % 500 == 0:
                    self.save(self.checkpointDIR,counter)

        print('\n\n\n so far so good !!')

    def __generator(self, InitInputSampleVector, Input_OutDegreeVector, trainFlag=True):
        """
        @porpose 实现 生戝器
        @生戝器架构: ---> 该生戝器 是一个 坝坷积网络
                RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu
                        -> generatorFC -> batch_norm_1 -> relu
                            -> batch_norm -> deconv -> batch_norm_2 -> relu
                                -> deconv -> batch_norm3 -> sigmoid -> generated_Adj-Matrix

        @InitInputSampleVector : 放入 生戝器 中的 一个 隝机的 初始坑針
        @Input_OutDegreeVector : 放入 生戝器 中的 一个 Adj-Matrix的出度元素列表坑針
        @trainFlag             : 是坦需覝进行训练(当仅仅为采样时并丝需覝训练)
        """
        with tf.variable_scope("generator") as scope:
            if trainFlag is False:
                scope.reuse_variables()

            # 1. 定义坷积层的图僝尺寸
            Height, Width = self.outputMat_H, self.outputMat_W
            deconv1_Height, deconv1_Width = int(Height/2), int(Width/2)
            deconv2_Height, deconv2_Weight = int(deconv1_Height/2), int(deconv1_Width/2)

            #  2.  RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu := h0 -> h0+OutDegreeVector
            """p.s. 下面所有的所谓"坑針"其实都是一个矩阵，其第一个维度就是batch_size"""
            ## 2.1 将隝机 初始的坑針 和 出度元素列表坑針 进行拼接
            input = tf.concat([InitInputSampleVector, Input_OutDegreeVector], axis=1) # <- input 为一个 100+100 共 200维 的坑針
            if debugFlag is True:
                if trainFlag is True:
                    print('============================= generator =============================')
                elif trainFlag is False:
                    print('============================= sampler =============================')
                print('input shape: ',input.shape) # (64, 200)
            ## 2.2 绝过全连接层->进行坑針化->通过激活函数relu
            h0=tf.nn.relu(
                self.generator_batch_norm_0(
                    linear(input_=input, output_size=self.generatorFC, scope="g_h0_lin"),
                    train = trainFlag
                )
                ) # <- h0 为一个 1024维 的坑針
            if debugFlag is True:
                print('h0 shape: ', h0.shape) # (64, 1024)
            ## 2.3 冝将Input_OutDegreeVector放在坎面
            h0 = tf.concat([h0, Input_OutDegreeVector], axis=1) # <- 添加h0 为一个 1024维 的坑針+ 100 维的那个OutDegreeVector [此时就坘戝了 64 * 1124 的矩阵]
            if debugFlag is True:
                print('h0+outdegree shape: ', h0.shape) # (64, 1124)

            # 3. h0+OutDegreeVector -> generatorFC -> batch_norm_1 -> relu := h1 -> reshape(h1) -> h1+OutDegreeVector
            output_size = self.generatorFilter*2 * deconv2_Height * deconv2_Weight  # <- 进一步增大维度，此时的维度应该是原“第二次坷积”之坎的维度。
                                                                                    #  --- self.generatorFilter*2 表示CNN网络所有的坷积filter多少
                                                                                    #  --- deconv2_Height * deconv2_Width 表示CNN网络第二次坷积之坎一共的个数
                                                                                    #  在这里，有 deconv2_Height = deconv2_Width = 100/4 = 25
                                                                                    #           self.generatorFilter*2 = 64*2 = 128
                                                                                    #           表明，线性化的长度应该是 output_size = 128*25*25 = 80000
                                                                                    # ∴ h1.shape = [64, 80000]
            h1 = tf.nn.relu(
                self.generator_batch_norm_1(
                    linear(input_=h0, output_size=output_size, scope="g_h1_lin"),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('h1 shape: ', h1.shape) # (64, 80000)
            #  由于此时h1为一个坑針，所以下面将这个坑針reshape戝一个Tensor，以酝坈（第二次）坷积的结果
            h1 = tf.reshape(h1, [self.batch_size, deconv2_Height, deconv2_Weight, self.generatorFilter*2])
            if debugFlag is True:
                print('reshape h1: ', h1.shape) # (64, 25, 25, 128)
            # 此时将reshape好的h1与之剝的OutDegreeVector进行坠加。
            # 这里需覝注愝的是，由于之剝OutDegreeVector仅仅是一个 64*100 的坑針，而这里 h1代表的是一个Tensor，所以应该先将OutDegreeVector扩展戝一个Tensor，冝进行坠加
            # 由于h1.shape = [64, 25, 25, 128], 所以对应的OutDegreeTensor应该是毝一个都有~所以应该写戝下面这个形弝
            OutDegreeTensor = tf.reshape(self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('reshape OutDegreeVector: ', OutDegreeTensor.shape)
            # 将OutDegreeTensor放在坎面
            h1 = conv_cond_concat(x=h1, y=OutDegreeTensor)
            if debugFlag is True:
                print('reshape h1 + OutDegreeVector: ', h1.shape) # (64, 25, 25, 128+100)

            # 4. h1+OutDegreeVector -> deconv2d -> batch_norm_2 -> relu := h2 -> h2+OutDegreeVector
            h2 = tf.nn.relu(
                self.generator_batch_norm_2(
                    deconv2d(input_=h1, output_shape=[self.batch_size, deconv1_Height, deconv1_Width, self.generatorFilter*2], name="g_h2"),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('deconv h2: ', h2.shape) # (64, 50, 50, 128)
            # 将OutDegreeTensor放在坎面
            h2 = conv_cond_concat(x=h2, y=OutDegreeTensor)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape)

            # 5. h2+OutDegreeVector -> deconv2d -> sigmoid # 注愝最坎一层没有坚batch_norm!!!
            h3 = tf.nn.sigmoid(
                deconv2d(input_=h2, output_shape=[self.batch_size, Height, Width, 1], name="g_h3")
            )
            if debugFlag is True:
                print('final layer h3 shape: ', h3.shape)

            return h3

    def __discriminator(self, InputAdjMatrix, Input_OutDegreeVector, reuse_variables=False):
        """
        @porpose 实现 带条件的 判别器 (其中, “条件”是指 毝次坷积的时候需覝带上该Matrix的OutDegreeVector信杯，这样坚的期望是生戝 基于该OutDegreeVector的Matrix)
        @生戝器架构: ---> 该判别器 是一个 坷积网络，其中，采用stride来代替 Max Pooling~
                    InputAdjMatrix+outDegreeVector -> condition(OutDegreeVector) & conv2d -> leaky_relu
                        -> condition(OutDegreeVector) & conv2d -> batch_norm_1 -> leaky_relu -> linear
                            -> FC -> batch_norm_2 -> leaky_relu
                                -> map_to_One_value -> sigmoid

        @InputAdjMatrix         : 放入 判别器 中的 一个 邻接矩阵
        @Input_OutDegreeVector  : 放入 判别器 中的 一个 Adj-Matrix的出度元素列表坑針
        @reuse_variables        : 判别器是坦需覝針用坘針~ ---TF
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse_variables is True:
                scope.reuse_variables() # 如果需覝reuse，那么下面的所有name都会在TF中被針新使用~~

            if debugFlag is True:
                print('============================= discriminator =============================')
                print('input shape: ',InputAdjMatrix.shape) # (64, 100, 100, 1)
            # 1. 首先将OutDegreeVector 坘戝Tensor 并于输入邻接矩阵拼接在一起
            OutDegreeTensor = tf.reshape(tensor=self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('outdegree tensor: ', OutDegreeTensor.shape) # (64, 1,1, 100)
            input_ = conv_cond_concat(x=InputAdjMatrix, y=OutDegreeTensor) # (64, 100,100, 100+1)
            if debugFlag is True:
                print('input+outdegree vector shape: ',input_.shape)

            """ 注愝: 这里是条件的坷积~
                所以output_filter一定覝加上最坎的OutDegree的结果~
            """
            # 2. 绝过第一个Hidden_Layer: InputAdjMatrix+outDegreeVector -> condition(OutDegreeVector) & conv2d -> leaky_relu := h0 -> h0+outDegreeVector
            h0 = lrelu(conv2d(input_=input_, output_filter=1+self.OutDegreeLength, name="d_h0_conv")) # 条件坷积!!!
            if debugFlag is True:
                print('CONDITION h0 shape: ', h0.shape) # (64, 50,50, 101)
            h0 = conv_cond_concat(x=h0,y=OutDegreeTensor)
            if debugFlag is True:
                print('h0+OutDegreeVector: ', h0.shape) # (64, 50,50, 201)

            # 3. 绝过第二个Hidden_Layer: h0+outDegreeVector -> condition(OutDegreeVector) & conv2d -> batch_norm_1 -> leaky_relu -> linear := h1
            h1 = lrelu(self.discriminator_batch_norm1(
                conv2d(input_=h0, output_filter = self.discriminatorFilter + self.OutDegreeLength, name="d_h1_conv1")
            )
                       )
            if debugFlag is True:
                print('CONDITION h1 shape: ', h1.shape) # (64, 25,25, 64+100)
            ## 扝平化, 以适用于坎面的全连接
            h1 = tf.reshape(tensor=h1, shape=[self.batch_size, -1])
            if debugFlag is True:
                print('linear h1: ', h1.shape) # (64, 25*25*164=102500)
            ## 冝加上OutDegreeVector(注愝, 此时由于扝平化了，所以应该采用outdegreeVector~~而非outDegreeTensor)
            h1 = tf.concat(values=[h1,self.OutDegreeVector],axis=1)
            if debugFlag is True:
                print('h1+OutDegreeVector: ', h1.shape) # (64, 102600)

            # 4. 绝过全连接层: h1 -> FC -> batch_norm_2 -> leaky_relu := h2
            h2 = lrelu(self.discriminator_batch_norm2(linear(input_=h1, output_size=self.discriminatorFC, scope="d_h2_lin")))
            if debugFlag is True:
                print('FC h2 shape: ', h2.shape) # (64, 1024)
            ## 冝加上OutDegreeVector
            h2 = tf.concat(values=[h2,self.OutDegreeVector], axis=1) # (64, 1124)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (64, 1124)

            # 5. 转杢戝 真/均, 并采用sigmoid函数映射到[0,1]上 : h3 -> map_to_One_value -> sigmoid
            h3 = linear(input_=h2, output_size=1, scope="d_h3_lin")
            if debugFlag is True:
                print('h3 map to ONE Value: ', h3.shape)
            h3_sigmoid = tf.nn.sigmoid(h3,name='sigmoid_h3')
            if debugFlag is True:
                print('h3 to sigmoid: ', h3_sigmoid.shape)

            return h3_sigmoid, h3

    def __load_AdjMatInfo(self, AdjMat_OutDegree_Dir, startPoint=0, endPoint=50):
        print('current Data DIR: ', AdjMat_OutDegree_Dir)
        adj_file = glob.glob('./data/'+AdjMat_OutDegree_Dir+'*.graph')[0]
        if debugFlag is True:
            print('current adj file dir: ', adj_file)
        degree_file = glob.glob('./data/'+AdjMat_OutDegree_Dir+'*.degree')[0]
        # 读入adj
        adj = pickle.load(open(adj_file,'rb'))
        degree = pickle.load(open(degree_file,'rb'))
        # 将adj字典转杢戝Tensor
        for i in range(startPoint, endPoint):
            if i == 0:
                Tensor = tf.stack([adj[i]],axis=0)
                DegreeTensor = tf.stack([degree[i]],axis=0)
            else:
                slice = tf.stack([adj[i]],axis=0)
                slice_degree = tf.stack([degree[i]],axis=0)
                Tensor = tf.concat([Tensor, slice],axis=0)
                DegreeTensor = tf.concat([DegreeTensor, slice_degree],axis=0)

        AdjMatTensor = tf.expand_dims(input=Tensor, axis=-1)
        if debugFlag is True:
            print(AdjMatTensor.shape)
            print(DegreeTensor.shape)

        return AdjMatTensor.eval(),DegreeTensor.eval()
    # ================================================
    # FROM DCGAN
    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size, self.outputMat_H, self.outputMat_W)

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



