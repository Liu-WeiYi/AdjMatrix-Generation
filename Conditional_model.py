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
            Momentum:           采用ADAM算法时所需要的Momentum的值 [0.5] --- based on DCGAN
            batch_size:         每一批读取的adj-mat大小 [10]
            generatorFilter:    生成器 初始的filter数值 [50] --- 注: 算法中每一层的filter都设置为该值的2倍 based on DCGAN
            discriminatorFilter:判别器 初始的filter数值 [50] --- 注: 算法中每一层的filter都设置为该值的2倍 based on DCGAN
            generatorFC:        生成器的全连接层的神经元个数 [1024]
            discriminatorFC:    判别器的全连接层的神经元个数 [1024]
            trainable_data_size:训练数据的总个数 [20000]
            inputMat_H:         输入邻接矩阵的Height [28] --- 注: 为避免卷积的时候出现小数，因此这里建议设置成4的整数倍，原因在于我们需要进行2次卷积操作，每次stride=2，所以会比原图像缩小4倍
            inputMat_W:         输入邻接矩阵的Width [28] --- 注: 同上
            outputMat_H:        输出邻接矩阵的Height [28] --- 注: 同上
            outputMat_W:        输出邻接矩阵的Width [28] --- 注: 同上
            OutDegree_Length:   当前 AdjMatrix的 出度向量长度，表示的是类别 [28]  --- 相当于DCGAN中的 y_lim [手写数字中的类别信息~]
            InitGen_Length:     用作 GEN的输入向量，最好与当前 AdjMatrix的 inputMat大小 保持一致 [28] --- 相当于DCGAN中的 z_lim
            inputPartitionDIR:  分割后的矩阵的存档点 [WS_test] --- 注: 最好与 dataset_name 保持一致，只不过这里指的是当前dataset_name所在的folder
            checkpointDIR:      存档点 地址 [checkpoint]
            sampleDIR:          采样得到的网络 输出地址 [samples]
            reconstructDIR:     重构网络的存放地址 [reconstruction]
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
        self.trainable_data_size = trainable_data_size
        self.inputMat_H          = inputMat_H
        self.inputMat_W          = inputMat_W
        self.outputMat_H         = outputMat_H
        self.outputMat_W         = outputMat_W
        self.OutDegreeLength     = OutDegree_Length
        # 用作Generator的输入，生成 当前网络
        self.InitSampleLength    = InitGen_Length
        # 指定 路径
        self.inputPartitionDIR   = inputPartitionDIR
        self.checkpointDIR       = checkpointDIR
        self.sampleDIR           = sampleDIR
        self.reconstructDIR      = reconstructDIR
        # 用作生成 拓扑结构 的方式
        self.link_possibility    = link_possibility

        """
        因为后面多个地方都需要用到 batch_norm 操作，因此在这里事先进行定义 --- based on DCGAN
        """
        self.generator_batch_norm_0 = batch_norm(name="g_bn0_%d"%self.inputMat_H)
        self.generator_batch_norm_1 = batch_norm(name="g_bn1_%d"%self.inputMat_H)
        self.generator_batch_norm_2 = batch_norm(name="g_bn2_%d"%self.inputMat_H)

        self.discriminator_batch_norm1 = batch_norm(name="d_bn1_%d"%self.inputMat_H)
        self.discriminator_batch_norm2 = batch_norm(name="d_bn2_%d"%self.inputMat_H)

        # 构建 GAN~
        self.modelConstrunction()

    def modelConstrunction(self):
        print('\n============================================================================')
        print('Model Construction ...')
        print('============================================================================')
        # 1. place holder 装输入的数据
        ## 用作存储 真实数据
        self.OutDegreeVector = tf.placeholder(tf.float32, shape=[self.batch_size, self.OutDegreeLength], name="Out_Degree_Vector_%d"%self.inputMat_H)
        self.inputMat = tf.placeholder(tf.float32, shape=[self.batch_size, self.inputMat_H, self.inputMat_W, 1], name="Real_Input_Adj_Matrix_%d"%self.inputMat_H)

        ## 用作存储 随机采样的属于，用于Generator生成器的输入
        self.InitSampleVector = tf.placeholder(tf.float32, shape=[None, self.InitSampleLength], name="GEN_Input_%d"%self.inputMat_H)

        # 2. Generator
        self.Generator = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector, trainFlag=True)

        # 3. Sampler --- 用作 生成图片的时候~~
        self.Sampler = self.__generator(InitInputSampleVector=self.InitSampleVector, Input_OutDegreeVector = self.OutDegreeVector, trainFlag=False)

        # 4. Reconstruction --- 用于 重建网络时
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
        ## 努力让真实数据全被识别为 正样本 --- ∴ 为tf.ones()
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real,labels=tf.ones_like(self.Discriminator_real))
        )
        ## 努力让Gen生成的数据全被识别 负样本 --- ∴ 为tf.zeros()
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake,labels=tf.zeros_like(self.Discriminator_fake))
        )
        self.discriminator_loss = self.d_loss_real + self.d_loss_fake

        # 7. Generator LOSS FUNCTION
        """ 根据公式~~
        注意 生成器的Loss Function应该算Discriminator的LOGIT哈＝＝
        """
        # 即努力让Gen生成的数据向 正样本 前进 ^_^ --- ∴ 为tf.ones()
        self.generator_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Discriminator_fake,labels=tf.ones_like(self.D_logits_fake))
        )

        # --> 放入TensorBoard中
        self.InitSampleVector_sum = tf.summary.histogram(name="GEN Input_%d"%self.inputMat_H,values=self.InitSampleVector)
        self.Generator_sum = tf.summary.histogram(name="GEN Output_%d"%self.inputMat_H,values=self.Generator)
        self.Discriminator_real_sum = tf.summary.histogram(name="Discriminator Real Output_%d"%self.inputMat_H, values=self.Discriminator_real)
        self.Discriminator_fake_sum = tf.summary.histogram(name="Discriminator Fake Output_%d"%self.inputMat_H, values=self.Discriminator_fake)

        self.d_loss_real_sum = tf.summary.scalar(name="Discriminator Real Loss_%d"%self.inputMat_H, tensor=self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar(name="Discriminator Fake Loss_%d"%self.inputMat_H, tensor=self.d_loss_fake)
        self.discriminator_loss_sum = tf.summary.scalar(name="Discriminator Loss_%d"%self.inputMat_H, tensor=self.discriminator_loss)

        self.generator_loss_sum = tf.summary.scalar(name="Generator Loss_%d"%self.inputMat_H,tensor=self.generator_loss)

        # --> 存储所有数据 --- based on DCGAN
        t_vars = tf.trainable_variables()

        self.discriminator_vars = [var for var in t_vars if "d_" in var.name]
        self.generator_vars = [var for var in t_vars if "g_" in var.name]

        # 存储checkpoint时用 --- based on DCGAN
        self.saver = tf.train.Saver()

        print('down ...')

    def train(self,returnFlag=False):
        # 如果需要return，则将returnFlag=True, 此时会返回做完所有epoch之后，根据每一块重新拼接起的网络
        print('\n============================================================================')
        print('Train Begin...')
        print('============================================================================')
        # 1. 定义 判别器 和 生成器的 Loss  Function 优化方法
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

        # 4. 准备投放训练数据~
        ## 读入 mat & Outdegree
        """
        如果内存不够~ 所以在后面再读~
        这里没有必要一次性读入
        """
        # data_mat, data_degree = self.__load_AdjMatInfo(AdjMat_OutDegree_Dir = self.inputPartitionDIR,startPoint=0,endPoint=20000,MatSize=self.inputMat_H)

        # 5. 记录checkpoint --- based on DCGAN
        could_load, checkpoint_counter = self.load(self.checkpointDIR)
        if could_load is True:
            counter = checkpoint_counter
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # 6. 训练开始~
        counter = 0
        start_time = time.time() # 记录当前开始时间~~
        for epoch in range(self.epoch):
            # step = data_mat.shape[0] // self.batch_size
            step  = self.trainable_data_size // self.batch_size

            for idx in range(0,step):
                ## 1. 读入所有数据 --- 由于一次读入有点大。。。这边分批读入 囧~~
                # batch_data_mat          = data_mat[idx*self.batch_size:(idx+1)*self.batch_size]
                # batch_data_degree       = data_degree[idx*self.batch_size:(idx+1)*self.batch_size]
                batch_data_mat, batch_data_degree = self.__load_AdjMatInfo( AdjMat_OutDegree_Dir = self.inputPartitionDIR,
                                                                            startPoint=idx*self.batch_size,
                                                                            endPoint=(idx+1)*self.batch_size,
                                                                            MatSize=self.inputMat_H)
                ## 1. 生成用于Gen的输入数据
                batch_generator_input   = np.random.uniform(low=-1,high=1,size=[self.batch_size, self.InitSampleLength]).astype(np.float32)

                ## 2. Update Discriminator 判别器
                _, summary_str = self.sess.run( [d_optimizer, self.discriminator_related_sum],
                                                feed_dict={
                                                    self.inputMat: batch_data_mat,
                                                    self.InitSampleVector : batch_generator_input,
                                                    self.OutDegreeVector : batch_data_degree
                                                })
                self.writer.add_summary(summary_str, counter)

                ## 3. Update Generator 生成器
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

                ## 5. 记录每次的loss --- based on DCGAN
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
                ### 1. 每次都输出 Loss Value
                counter += 1
                print("Epoch: [%d] | [%d/%d] time: %4.4f, d_loss: %.8f, g_loss: %.8f"%(epoch, idx, step, time.time()-start_time, errD_fake+errD_real, errG))
                ### 2. 每500次生成一张图片。。。并保存~
                if counter % 500 == 0:
                    ## 用作Generator输入 ---  随机生成一个具有Self.InitSampleLenght长度的向量，利用GAN生成一个 Adj-Mat
                    sample_input = np.random.uniform(low=-1, high=1, size=[self.batch_size, self.InitSampleLength])
                    if debugFlag is True:
                        print('sampled input for Generator shape: ', sample_input.shape) # (20, 28)
                    # sample_mat = data_mat[0:self.batch_size]
                    # sample_labels = data_degree[0:self.batch_size]
                    # 数据采用分段载入方式, 读入真实数据是为了计算 Loss 值
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

                if counter % 500 == 0:
                    # checkpointDIR = self.checkpointDIR
                    self.save(self.checkpointDIR,counter)

    def saveModel(self):
        """
        存储训练好的模型
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

        # step.0 重新构建每一块
        each_part = {}
        for i in range(self.trainable_data_size):
            # for generator input~ 因为是 一个一个恢复， 所以下面 sample_input 的size 中第一个代表batch_size的元素应该是 1
            sample_input = np.random.uniform(low=-1, high=1.0, size=[1, self.InitSampleLength])

            # 读入每一个真实块的数据，只是为了计算 Loss 值。其实:
            """
            1. 只需要楼上 sample_input       就可以让Generator生成相应的 Adj_Mat~
            2. 再加上楼下 partition_labels   就可以让Generator生成某一类的 Adj_Mat~
            """
            partition_mat,partition_labels = self.__load_AdjMatInfo(AdjMat_OutDegree_Dir = self.inputPartitionDIR,
                                                                    startPoint=i,
                                                                    endPoint=i+1,
                                                                    MatSize=self.inputMat_H)

            #reconstruct_adj, d_loss, g_loss = self.sess.run([self.re_Construction, self.discriminator_loss, self.generator_loss],
                                                            #feed_dict={
                                                                #self.InitSampleVector : sample_input,
                                                                #self.re_Mat : partition_mat,
                                                                #self.re_OutDegreeVector : partition_labels
                                                            #})
            reconstruct_adj = self.sess.run([self.re_Construction],
                                            feed_dict={
                                                self.InitSampleVector : sample_input,
                                                self.re_Mat : partition_mat,
                                                self.re_OutDegreeVector : partition_labels
                                            })

            each_part[i] = np.squeeze(reconstruct_adj[0])

        # step.1  采用 重构出的reconstruct_adj 以及 每一块的映射文件 <filename>_MatSize.map (part2Node) 拼接所有块
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
        @porpose 实现 生成器
        @生成器架构: ---> 该生成器 是一个 反卷积网络
                RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu
                        -> generatorFC -> batch_norm_1 -> relu
                            -> batch_norm -> deconv -> batch_norm_2 -> relu
                                -> deconv -> batch_norm3 -> sigmoid -> generated_Adj-Matrix

        @InitInputSampleVector : 放入 生成器 中的 一个 随机的 初始 向量
        @Input_OutDegreeVector : 放入 生成器 中的 一个 Adj-Matrix的 出度元素列表 向量
        @trainFlag             : 是否需要进行训练 (当仅仅为采样时不需要训练)
        """
        with tf.variable_scope("generator") as scope:
            if trainFlag is False:
                scope.reuse_variables()

            # 1. 定义 所有反卷积层  尺寸 --- 尺寸与卷积层的保持一致
            Height, Width = self.outputMat_H, self.outputMat_W
            """p.s. 为保证/4能够除尽，这里引入 math.ceil对结果进行向上取整"""
            deconv1_Height, deconv1_Width = int(math.ceil(Height/2)), int(math.ceil(Width/2))
            deconv2_Height, deconv2_Weight = int(math.ceil(deconv1_Height/2)), int(math.ceil(deconv1_Width/2))

            #  2.  RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu := h0 -> h0+OutDegreeVector
            """p.s. 下面所有的所谓"向量"其实都是一个矩阵，其第一个维度就是batch_size"""
            ## 2.1 将随机 初始的 向量 和 出度元素列表 向量 进行拼接
            input = tf.concat([InitInputSampleVector, Input_OutDegreeVector], axis=1) # <- input 为一个 28+28 共 56 的 向量
            if debugFlag is True:
                if trainFlag is True:
                    print('============================= generator =============================')
                elif trainFlag is False:
                    print('============================= sampler =============================')
                print('input shape: ',input.shape) # (20, 56)
            ## 2.2 经过全连接层->进行 向量化->通过激活函数relu
            h0=tf.nn.relu(
                self.generator_batch_norm_0(
                    linear(input_=input, output_size=self.generatorFC, scope="g_h0_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
                ) # <- h0 为一个 1024维 的 向量
            if debugFlag is True:
                print('h0 shape: ', h0.shape) # (20, 1024)
            ## 2.3 将Input_OutDegreeVector放在后面
            h0 = tf.concat([h0, Input_OutDegreeVector], axis=1) # <- 添加h0 为一个 1024维 的向量 + 28 维的那个OutDegreeVector [此时就变成了 20 * 1052 的矩阵]
            if debugFlag is True:
                print('h0+outdegree shape: ', h0.shape) # (20, 1052)

            # 3. h0+OutDegreeVector -> generatorFC -> batch_norm_1 -> relu := h1 -> reshape(h1) -> h1+OutDegreeVector
            output_size = self.generatorFilter*2 * deconv2_Height * deconv2_Weight  # <- 进一步增大维度，此时的维度应该是原“第二次卷积”之后的维度。
                                                                                    #  --- self.generatorFilter*2 表示CNN网络所有的卷积filter多少
                                                                                    #  --- deconv2_Height * deconv2_Width 表示CNN网络第二次卷积之后一共的个数
                                                                                    #  在这里，有 deconv2_Height = deconv2_Width = 28/4 = 7
                                                                                    #           self.generatorFilter*2 = 50*2 = 100
                                                                                    #           表明，线性化的长度应该是 output_size = 100*7*7 = 4900
                                                                                    # ∴ h1.shape = [20, 4900]
            h1 = tf.nn.relu(
                self.generator_batch_norm_1(
                    linear(input_=h0, output_size=output_size, scope="g_h1_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('h1 shape: ', h1.shape) # (20, 4900)
            #  由于此时h1为一个 向量，所以下面将这个 向量reshape成一个Tensor
            h1 = tf.reshape(h1, [self.batch_size, deconv2_Height, deconv2_Weight, self.generatorFilter*2])
            if debugFlag is True:
                print('reshape h1: ', h1.shape) # (20, 7,7, 100)
            # 此时将reshape好的h1与之前的OutDegreeVector进行追加。
            # 这里需要注意的是，由于之前OutDegreeVector仅仅是一个 20*28 的 向量，而这里 h1代表的是一个Tensor，所以应该先将OutDegreeVector扩展成一个Tensor，再进行追加
            # 由于h1.shape = [20, 7, 7, 100], 所以对应的OutDegreeTensor应该是每一个都有~所以应该写成下面这个形式
            OutDegreeTensor = tf.reshape(self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('reshape OutDegreeVector: ', OutDegreeTensor.shape) # (20, 1, 1, 28)
            # 将OutDegreeTensor放在后面
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
            # 将OutDegreeTensor放在后面
            h2 = conv_cond_concat(x=h2, y=OutDegreeTensor)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (20, 14, 14, 128)

            # 5. h2+OutDegreeVector -> deconv2d -> sigmoid # 注意最后一层没有batch_norm!!!
            h3 = tf.nn.sigmoid(
                deconv2d(input_=h2, output_shape=[self.batch_size, Height, Width, 1], name="g_h3_%d"%self.inputMat_H)
            )
            if debugFlag is True:
                print('final layer h3 shape: ', h3.shape) # (20, 28, 28, 1)

            return h3

    def __discriminator(self, InputAdjMatrix, Input_OutDegreeVector, reuse_variables=False):
        """
        @porpose 实现 带条件的 判别器 (其中, “条件”是指 每次卷积的时候需要带上该Matrix的OutDegreeVector条件，这样做的期望是生成 基于该OutDegreeVector的Matrix)
        @生成器架构: ---> 该判别器 是一个 卷积网络，其中，采用stride来代替 Max Pooling~  --- based on DCGAN
                    InputAdjMatrix+outDegreeVector -> condition(OutDegreeVector) & conv2d -> leaky_relu
                        -> condition(OutDegreeVector) & conv2d -> batch_norm_1 -> leaky_relu -> linear
                            -> FC -> batch_norm_2 -> leaky_relu
                                -> map_to_One_value -> sigmoid

        @InputAdjMatrix         : 放入 判别器 中的 一个 邻接矩阵
        @Input_OutDegreeVector  : 放入 判别器 中的 一个 Adj-Matrix的出度元素列表 向量
        @reuse_variables        : 判别器是否需要复用数据~ 这是因为D会有两个，而这两个其实都指向的是一个D~ ---TF
        """
        with tf.variable_scope("discriminator") as scope:
            if reuse_variables is True:
                scope.reuse_variables() # 如果需要reuse，那么下面的所有name都会在TF中被复新使用~~

            if debugFlag is True:
                print('============================= discriminator =============================')
                print('input shape: ',InputAdjMatrix.shape) # (20, 28, 28, 1)
            # 1. 首先将OutDegreeVector 转成Tensor 并于输入邻接矩阵拼接在一起
            OutDegreeTensor = tf.reshape(tensor=self.OutDegreeVector, shape=[self.batch_size, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('outdegree tensor: ', OutDegreeTensor.shape) # (20, 1,1, 28)
            input_ = conv_cond_concat(x=InputAdjMatrix, y=OutDegreeTensor) # (20, 28,28, 1+28)
            if debugFlag is True:
                print('input+outdegree vector shape: ',input_.shape) # (20, 28,28, 1+28)

            """ 注意: 这里是条件的卷积~
                所以output_filter一定添加上最后的OutDegree的结果~
            """
            # 2. 经过第一个Hidden_Layer: InputAdjMatrix+outDegreeVector -> condition(OutDegreeVector) & conv2d -> leaky_relu := h0 -> h0+outDegreeVector
            h0 = lrelu(conv2d(input_=input_, output_filter=1+self.OutDegreeLength, name="d_h0_conv_%d"%self.inputMat_H)) # 条件卷积!!!
            if debugFlag is True:
                print('CONDITION h0 shape: ', h0.shape) # (20, 14,14, 29)
            h0 = conv_cond_concat(x=h0,y=OutDegreeTensor)
            if debugFlag is True:
                print('h0+OutDegreeVector: ', h0.shape) # (20, 14,14, 29+28=57)

            # 3. 经过第二个Hidden_Layer: h0+outDegreeVector -> condition(OutDegreeVector) & conv2d -> batch_norm_1 -> leaky_relu -> linear := h1
            h1 = lrelu(self.discriminator_batch_norm1(
                conv2d(input_=h0, output_filter = self.discriminatorFilter + self.OutDegreeLength, name="d_h1_conv1_%d"%self.inputMat_H)
                                            )
                        )
            if debugFlag is True:
                print('CONDITION h1 shape: ', h1.shape) # (20, 7,7, 50+28)
            ## 扁平化, 以适用于后面的全连接
            h1 = tf.reshape(tensor=h1, shape=[self.batch_size, -1])
            if debugFlag is True:
                print('linear h1: ', h1.shape) # (20, 7*7*78=3822)
            ## 再加上OutDegreeVector(注意, 此时由于扝平化了，所以应该采用outdegreeVector~~而非outDegreeTensor)
            h1 = tf.concat(values=[h1,self.OutDegreeVector],axis=1)
            if debugFlag is True:
                print('h1+OutDegreeVector: ', h1.shape) # (20, 3822+28=3850)

            # 4. 经过全连接层: h1 -> FC -> batch_norm_2 -> leaky_relu := h2
            h2 = lrelu(self.discriminator_batch_norm2(linear(input_=h1, output_size=self.discriminatorFC, scope="d_h2_lin_%d"%self.inputMat_H)))
            if debugFlag is True:
                print('FC h2 shape: ', h2.shape) # (20, 1024)
            ## 再加上OutDegreeVector
            h2 = tf.concat(values=[h2,self.OutDegreeVector], axis=1) # (20, 1024+28=1052)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (20, 1052)

            # 5. 转换成 真/均, 并采用sigmoid函数映射到[0,1]上 : h3 -> map_to_One_value -> sigmoid
            h3 = linear(input_=h2, output_size=1, scope="d_h3_lin_%d"%self.inputMat_H)
            if debugFlag is True:
                print('h3 map to ONE Value: ', h3.shape) # (20,1)
            h3_sigmoid = tf.nn.sigmoid(h3,name="sigmoid_h3_%d"%self.inputMat_H)
            if debugFlag is True:
                print('h3 to sigmoid: ', h3_sigmoid.shape) # (20, 1)

            return h3_sigmoid, h3

    def __re_Construction(self, InitInputSampleVector, Input_OutDegreeVector, trainFlag=False):
        """
        @porpose 实现 生成器
        @生成器架构: ---> 该生成器 是一个 反卷积网络
                RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu
                        -> generatorFC -> batch_norm_1 -> relu
                            -> batch_norm -> deconv -> batch_norm_2 -> relu
                                -> deconv -> batch_norm3 -> sigmoid -> generated_Adj-Matrix

        @InitInputSampleVector : 放入 生成器 中的 一个 随机的 初始 向量
        @Input_OutDegreeVector : 放入 生成器 中的 一个 Adj-Matrix的 出度元素列表 向量
        @trainFlag             : 是否需要进行训练 (当仅仅为采样时不需要训练)
        """
        with tf.variable_scope("generator") as scope:
            if trainFlag is False:
                scope.reuse_variables()

            # 1. 定义 所有反卷积层  尺寸 --- 尺寸与卷积层的保持一致
            Height, Width = self.outputMat_H, self.outputMat_W
            """p.s. 为保证/4能够除尽，这里引入 math.ceil对结果进行向上取整"""
            deconv1_Height, deconv1_Width = int(math.ceil(Height/2)), int(math.ceil(Width/2))
            deconv2_Height, deconv2_Weight = int(math.ceil(deconv1_Height/2)), int(math.ceil(deconv1_Width/2))

            #  2.  RandomValueVector+OutDegreeVector -> generatorFC -> batch_norm_0 -> relu := h0 -> h0+OutDegreeVector
            """p.s. 下面所有的所谓"向量"其实都是一个矩阵，其第一个维度就是batch_size"""
            ## 2.1 将随机 初始的 向量 和 出度元素列表 向量 进行拼接
            input = tf.concat([InitInputSampleVector, Input_OutDegreeVector], axis=1) # <- input 为一个 28+28 共 56 的 向量
            if debugFlag is True:
                if trainFlag is True:
                    print('============================= generator =============================')
                elif trainFlag is False:
                    print('============================= ReConstruction =============================')
                print('input shape: ',input.shape) # (20, 56)
            ## 2.2 经过全连接层->进行 向量化->通过激活函数relu
            h0=tf.nn.relu(
                self.generator_batch_norm_0(
                    linear(input_=input, output_size=self.generatorFC, scope="g_h0_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
                ) # <- h0 为一个 1024维 的 向量
            if debugFlag is True:
                print('h0 shape: ', h0.shape) # (20, 1024)
            ## 2.3 将Input_OutDegreeVector放在后面
            h0 = tf.concat([h0, Input_OutDegreeVector], axis=1) # <- 添加h0 为一个 1024维 的向量 + 28 维的那个OutDegreeVector [此时就变成了 20 * 1052 的矩阵]
            if debugFlag is True:
                print('h0+outdegree shape: ', h0.shape) # (20, 1052)

            # 3. h0+OutDegreeVector -> generatorFC -> batch_norm_1 -> relu := h1 -> reshape(h1) -> h1+OutDegreeVector
            output_size = self.generatorFilter*2 * deconv2_Height * deconv2_Weight  # <- 进一步增大维度，此时的维度应该是原“第二次卷积”之后的维度。
                                                                                    #  --- self.generatorFilter*2 表示CNN网络所有的卷积filter多少
                                                                                    #  --- deconv2_Height * deconv2_Width 表示CNN网络第二次卷积之后一共的个数
                                                                                    #  在这里，有 deconv2_Height = deconv2_Width = 28/4 = 7
                                                                                    #           self.generatorFilter*2 = 50*2 = 100
                                                                                    #           表明，线性化的长度应该是 output_size = 100*7*7 = 4900
                                                                                    # ∴ h1.shape = [20, 4900]
            h1 = tf.nn.relu(
                self.generator_batch_norm_1(
                    linear(input_=h0, output_size=output_size, scope="g_h1_lin_%d"%self.inputMat_H),
                    train = trainFlag
                )
            )
            if debugFlag is True:
                print('h1 shape: ', h1.shape) # (20, 4900)
            #  由于此时h1为一个 向量，所以下面将这个 向量reshape成一个Tensor
            h1 = tf.reshape(h1, [1, deconv2_Height, deconv2_Weight, self.generatorFilter*2])
            if debugFlag is True:
                print('reshape h1: ', h1.shape) # (20, 7,7, 100)
            # 此时将reshape好的h1与之前的OutDegreeVector进行追加。
            # 这里需要注意的是，由于之前OutDegreeVector仅仅是一个 20*28 的 向量，而这里 h1代表的是一个Tensor，所以应该先将OutDegreeVector扩展成一个Tensor，再进行追加
            # 由于h1.shape = [20, 7, 7, 100], 所以对应的OutDegreeTensor应该是每一个都有~所以应该写成下面这个形式
            OutDegreeTensor = tf.reshape(self.re_OutDegreeVector, shape=[1, 1, 1, self.OutDegreeLength])
            if debugFlag is True:
                print('reshape OutDegreeVector: ', OutDegreeTensor.shape) # (20, 1, 1, 28)
            # 将OutDegreeTensor放在后面
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
            # 将OutDegreeTensor放在后面
            h2 = conv_cond_concat(x=h2, y=OutDegreeTensor)
            if debugFlag is True:
                print('h2 + OutDegreeVector: ', h2.shape) # (20, 14, 14, 128)

            # 5. h2+OutDegreeVector -> deconv2d -> sigmoid # 注意最后一层没有batch_norm!!!
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

        # 读入adj
        adj = pickle.load(open(adj_file,'rb'))
        degree = pickle.load(open(degree_file,'rb'))
        if debugFlag is True:
            print('loaded adj file, adj.keys() = ', len(adj.keys()), end='\t')
            print('loaded degree file, degree.keys() = ', len(degree.keys()), end='\t')
            print('start point= ', startPoint, '\tend point= ', endPoint)
        # 将adj字典转换成Tensor
        for i in range(startPoint, endPoint):
            # if debugFlag is True:
            #     if i+1 % 5000 == 0:
            #         print('transfer %d Tensors'%i)
            if i == startPoint:
                if i not in adj.keys():
                    i = len(adj.keys())-1
                #Tensor = tf.stack([adj[i]],axis=0)
                Tensor = np.stack([adj[i]],axis=0)
                #DegreeTensor = tf.stack([degree[i]],axis=0)
                DegreeTensor = np.stack([degree[i]],axis=0)
            else:
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



