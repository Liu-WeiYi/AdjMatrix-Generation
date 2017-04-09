#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  主程序入口
            采用DCGAN思想 生成拓扑网络
  Created: 04/08/17
"""
import os
import tensorflow as tf
import numpy as np
import argparse

from model import *
from utils import *

# ------------------------------
# arg input
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # Current Net Name
    parser.add_argument("--Dataset",                type=str,   default="Facebook", help="Datasets Choose")

    # Model Hyper-parameters
    parser.add_argument("--epoch",                  type=int,   default=25,         help="training steps [25]")
    parser.add_argument("--learning_rate",          type=int,   default=0.0002,     help="Learning rate for adam [0.0002]")
    parser.add_argument("--Momentum_term_adam",     type=float, default=0.5,        help="Momentum term of adam [0.5]")
    parser.add_argument("--train_size",             type=float, default=np.inf,     help="Trainning Adj-Matrix Size [np.inf]")
    parser.add_argument("--batch_size",             type=int,   default=64,         help="The size of batch [64]")
    parser.add_argument("--generator_Filter",       type=int,   default=64,         help="generator filter size [64]")
    parser.add_argument("--discriminator_Filter",   type=int,   default=64,         help="discriminator filter size [64]")
    parser.add_argument("--generator_FC_length",    type=int,   default=1024,       help="generator fully connected layer length [1024]")
    parser.add_argument("--discriminator_FC_length",type=int,   default=1024,       help="discriminator fully connected layer length [1024]")

    # input/output Adj-Matrix
    parser.add_argument("--inputMat_Height",        type=int,   default=100,        help="input Adj-Matrix Height [100]")
    parser.add_argument("--inputMat_Width",         type=int,   default=100,        help="input Adj-Matrix Width [100]")
    parser.add_argument("--outputMat_Height",       type=int,   default=100,        help="output Adj-Matrix Height [100]")
    parser.add_argument("--outputMat_Width",        type=int,   default=100,        help="output Adj-Matrix Width [100]")

    # Out-Degree Vector
    parser.add_argument("--OutDegree_Length",       type=int,   default=100,        help="input degree vector length [100]")

    # Init Generator Sample Vector
    parser.add_argument("--InitGen_Length",         type=int,   default=100,        help="output degree vector length [100]")

    # tersorboard requirement
    parser.add_argument("--input_partition_dir",    type=str,   default="./graphs",       help="Directory Name to Input Partition Graphs")
    parser.add_argument("--checkpoint_dir",         type=str,   default="./checkpoint",   help="Directory Name to save checkpoints [./checkpoint]")
    parser.add_argument("--samples_dir",            type=str,   default="./samples",      help="Directory Name to save Samples Topology [./samples]")

    return parser.parse_args()

# ------------------------------
# main
# ------------------------------
def main(args):
  with tf.device('cpu:0'): # 强制在CPU上运行 囧~
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
      adjMatGen = adj_Matrix_Generator(
        sess=sess,
        dataset_name=args.Dataset,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        Momentum = args.Momentum_term_adam,
        train_size=args.train_size,
        batch_size=args.batch_size,
        generatorFilter=args.generator_Filter,
        discriminatorFilter=args.discriminator_Filter,
        generatorFC=args.generator_FC_length,
        discriminatorFC=args.discriminator_FC_length,
        inputMat_H=args.inputMat_Height,
        inputMat_W=args.inputMat_Width,
        outputMat_H=args.outputMat_Height,
        outputMat_W=args.outputMat_Width,
        OutDegree_Length = args.OutDegree_Length,
        InitGen_Length=args.InitGen_Length,
        inputPartitionDIR=args.input_partition_dir,
        checkpointDIR=args.checkpoint_dir,
        sampleDIR=args.samples_dir
      )

      show_all_variables() # TF中的所有变量

if __name__ == '__main__':
    main(parse_args())
    print('DOWN')