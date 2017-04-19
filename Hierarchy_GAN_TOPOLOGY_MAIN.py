#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Hierarchy GAN-Topology 生成器 主程序入口
            采用DCGAN思想 生成拓扑网络
  Created: 04/13/17
"""
import os
import tensorflow as tf
import numpy as np
import argparse

from Hierarchy_model import *
from utils import *

# ------------------------------
# arg input
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # For Hierarchy GAN itself
    parser.add_argument("--training_step",          type=int,   default=500,       help="specify training step for hierarchy gan~ [500]")
    parser.add_argument("--permutation_step",       type=int,   default=0,         help="specify how many times to permute the adj for each layer~ [0]")

    # Current Net Name
    parser.add_argument("--Dataset",                type=str,   default="facebook",  help="Datasets Choose")

    # Model Hyper-parameters
    parser.add_argument("--epoch",                  type=int,   default=1000,         help="training steps [20]")
    parser.add_argument("--learning_rate",          type=int,   default=0.0002,     help="Learning rate for adam [0.0002]")
    parser.add_argument("--Momentum_term_adam",     type=float, default=0.5,        help="Momentum term of adam [0.5]")
    parser.add_argument("--batch_size",             type=int,   default=5,         help="The size of batch [5]")
    parser.add_argument("--generator_Filter",       type=int,   default=3,         help="generator filter size [3]")
    parser.add_argument("--discriminator_Filter",   type=int,   default=3,         help="discriminator filter size [3]")
    parser.add_argument("--generator_FC_length",    type=int,   default=1024,       help="generator fully connected layer length [1024]")
    parser.add_argument("--discriminator_FC_length",type=int,   default=1024,       help="discriminator fully connected layer length [1024]")

    # input/output Adj-Matrix
    parser.add_argument("--training_info_dir",      type=str,   default="facebook_partition_info.pickle", help="specify trainable_data_size and inputMatSize")

    # Out-Degree Vector
    parser.add_argument("--OutDegree_Length",       type=int,   default=1,        help="input degree vector class label length [1]")

    # tersorboard requirement
    parser.add_argument("--input_partition_dir",    type=str,   default="facebook",         help="Directory Name to Input Partition Graphs")
    parser.add_argument("--checkpoint_dir",         type=str,   default="checkpoint",       help="Directory Name to save checkpoints [./checkpoint]")
    parser.add_argument("--samples_dir",            type=str,   default="condition_samples",help="Directory Name to save Samples Topology [./samples]")
    parser.add_argument("--reconstruction_dir",     type=str,   default="reconstruction",   help="Directory Name to save reconstructed Topology [reconstruction]")

    # adj-mat constuction possibility
    parser.add_argument("--link_possibility",       type=float, default=0.5,              help="if a value in sampled adj-mat is greater than link-possibility-threshold, then there is an edge between two nodes on positions [0.5]")

    return parser.parse_args()

# ------------------------------
# main
# ------------------------------
def main(args):
    dir = os.path.join('data',args.Dataset,'')
    if not os.path.isdir(dir):
        print('[!!!]current dir do not exist: ', dir)
        raise SystemExit('[!!!]Please create Data FIRST !!')

    with tf.device('cpu:0'): # 强制在CPU上运行 囧~
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            """首先判断是否存在已经训练好的结果*.nxgraph，如果没有，则需要调用Condition_GAN进行重新训练"""
            trained_path = os.path.join(args.reconstruction_dir,args.Dataset,"Hierarchy",'')
            trainedFlag = os.path.exists(trained_path)

            # 1. construct Model for each layer
            adjMatGen = Hierarchy_adjMatrix_Generator(
                sess=sess,
                dataset_name=args.Dataset, permutation_num=args.permutation_step,
                epoch=args.epoch,
                learning_rate=args.learning_rate,
                Momentum = args.Momentum_term_adam,
                batch_size=args.batch_size,
                generatorFilter=args.generator_Filter,
                discriminatorFilter=args.discriminator_Filter,
                generatorFC=args.generator_FC_length,
                discriminatorFC=args.discriminator_FC_length,
                training_info_dir=args.training_info_dir,
                OutDegree_Length = args.OutDegree_Length,
                inputPartitionDIR=args.input_partition_dir,
                checkpointDIR=args.checkpoint_dir,
                sampleDIR=args.samples_dir,
                reconstructDIR=args.reconstruction_dir,
                link_possibility=args.link_possibility,
                trainedFlag=trainedFlag
            )

            # 2. using weight for each layer, and construct Hierarchy Model
            adjMatGen.modelConstruction()

            # 3. train Hierarchy Model to get weight and constructed adj
            weight_list, reconstructed_Adj = adjMatGen.train(training_step=args.training_step)
            print("=======================\nreconstuction DOWN!\n=======================")
            if debugFlag is True:
                print('weight list: ', weight_list)
                print('reconstructed_adj shape: ', reconstructed_Adj.shape)

            # C.main("facebook","Hierarchy",reconstructed_Adj)


            # 4. save Model
            pickle.dump(reconstructed_Adj, open(trained_path+"reconstructed_%s.adj"%args.Dataset,'wb'))

if __name__ == '__main__':
    main(parse_args())
    print('DOWN')