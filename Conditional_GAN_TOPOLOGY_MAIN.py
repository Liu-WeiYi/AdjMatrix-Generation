#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Condition GAN-Topology 生成器 主程序入口
            采用DCGAN思想 生成拓扑网络
  Created: 04/08/17
"""
import os
import tensorflow as tf
import numpy as np
import argparse

from Conditional_model import *
from utils import *

# ------------------------------
# arg input
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Current Net Name
    parser.add_argument("--Dataset",                type=str,   default="facebook",  help="Datasets Choose")

    # Model Hyper-parameters
    parser.add_argument("--epoch",                  type=int,   default=1000,         help="training steps [20]")
    parser.add_argument("--learning_rate",          type=int,   default=0.0002,     help="Learning rate for adam [0.0002]")
    parser.add_argument("--Momentum_term_adam",     type=float, default=0.5,        help="Momentum term of adam [0.5]")
    parser.add_argument("--batch_size",             type=int,   default=20,         help="The size of batch [20]")
    parser.add_argument("--generator_Filter",       type=int,   default=3,         help="generator filter size [50]")
    parser.add_argument("--discriminator_Filter",   type=int,   default=3,         help="discriminator filter size [50]")
    parser.add_argument("--generator_FC_length",    type=int,   default=1024,       help="generator fully connected layer length [1024]")
    parser.add_argument("--discriminator_FC_length",type=int,   default=1024,       help="discriminator fully connected layer length [1024]")

    # input/output Adj-Matrix
    parser.add_argument("--trainable_data_size",    type=int,   default=404,     help="Training Data Total Number [20000]")
    parser.add_argument("--inputMat_Height",        type=int,   default=10,        help="input Adj-Matrix Height [28]")
    parser.add_argument("--inputMat_Width",         type=int,   default=10,        help="input Adj-Matrix Width [28]")
    parser.add_argument("--outputMat_Height",       type=int,   default=10,        help="output Adj-Matrix Height [28]")
    parser.add_argument("--outputMat_Width",        type=int,   default=10,        help="output Adj-Matrix Width [28]")

    # Out-Degree Vector
    parser.add_argument("--OutDegree_Length",       type=int,   default=10,        help="input degree vector length [28]")

    # Init Generator Sample Vector
    parser.add_argument("--InitGen_Length",         type=int,   default=10,        help="output degree vector length [28]")

    # tersorboard requirement
    parser.add_argument("--input_partition_dir",    type=str,   default="facebook",      help="Directory Name to Input Partition Graphs")
    parser.add_argument("--checkpoint_dir",         type=str,   default="checkpoint",   help="Directory Name to save checkpoints [./checkpoint]")
    parser.add_argument("--samples_dir",            type=str,   default="samples",      help="Directory Name to save Samples Topology [./samples]")

    # adj-mat constuction possibility
    parser.add_argument("--link_possibility",       type=float, default=0.5,              help="if a value in sampled adj-mat is greater than link-possibility-threshold, then there is an edge between two nodes on positions [0.5]")

    return parser.parse_args()

# ------------------------------
# main
# ------------------------------
def main(args):
    dir = './data/'+args.Dataset
    if not os.path.isdir(dir):
        print('[!!!]current dir do not exist: ', dir)
        raise SystemExit('[!!!]Please create Data FIRST !!')

    with tf.device('cpu:0'): # 强制在CPU上运行 囧~
        with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
            # 1. construct Model
            adjMatGen = Condition_adjMatrix_Generator(
                sess=sess,
                dataset_name=args.Dataset,
                epoch=args.epoch,
                learning_rate=args.learning_rate,
                Momentum = args.Momentum_term_adam,
                batch_size=args.batch_size,
                generatorFilter=args.generator_Filter,
                discriminatorFilter=args.discriminator_Filter,
                generatorFC=args.generator_FC_length,
                discriminatorFC=args.discriminator_FC_length,
                trainable_data_size=args.trainable_data_size,
                inputMat_H=args.inputMat_Height,
                inputMat_W=args.inputMat_Width,
                outputMat_H=args.outputMat_Height,
                outputMat_W=args.outputMat_Width,
                OutDegree_Length = args.OutDegree_Length,
                InitGen_Length=args.InitGen_Length,
                inputPartitionDIR=args.input_partition_dir,
                checkpointDIR=args.checkpoint_dir,
                sampleDIR=args.samples_dir,
                link_possibility=args.link_possibility
            )

            show_all_variables() # TF中的所有变量

            # 2. train Model
            adjMatGen.train()

            # 3. save Model
            save_path = adjMatGen.saveModel()
            print('Trained Model Path: ', save_path)

if __name__ == '__main__':
    main(parse_args())
    print('DOWN')