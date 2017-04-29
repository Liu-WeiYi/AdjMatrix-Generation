#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Multilayer Hierarchy GAN-Topology 生成器 主程序入口
            读入多个网络。。。生成给定Size的网络拓扑结构
  @Usage:   必须在建立了reconstruction/<filename>/Hierarchy_Multilayer/*.nxgraph的基础之上才能运行!!!
  Created: 04/22/17
"""

import os
import tensorflow as tf
import numpy as np
import argparse

from Hierarchy_model import *
from utils import *
from Hierarchy_model_Multilayer import *

import Compair as C


# ------------------------------
# arg input
# ------------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    # For Hierarchy GAN itself
    parser.add_argument("--training_step",          type=int,   default=500,              help="specify training step for hierarchy gan")
    # Current Net Name
    parser.add_argument("--Datasets",               type=str,   nargs="+",   default=["facebook", "facebook2"],   help="Datasets Choose")
    # Desired Generated Net Size (期望生成的网络大小)
    parser.add_argument("--Desired_graph_size",     type=int,   default=4000,             help="Desired Generated Net Size" )

    parser.add_argument("--reconstruction_dir",     type=str,   default="reconstruction", help="Directory Name to save reconstructed Topology [reconstruction]")

    return parser.parse_args()

# ------------------------------
# main
# ------------------------------
def main(args):
    for dataset in args.Datasets:
        dir = os.path.join('data',dataset,'')
        if not os.path.isdir(dir):
            print('[!!!]current dir do not exist: ', dir)
            raise SystemExit('[!!!]Please create Data FIRST !!')
        reconstructed_dir = os.path.join("reconstruction", dataset, "Hierarchy", "")
        if not os.path.isdir(reconstructed_dir) or glob.glob(reconstructed_dir) == []:
            print('[!!!]current dir or reconstructed *.nxgraph do not exist: ', reconstructed_dir)
            raise SystemExit('[!!!]Please create Data FIRST !!')

    with tf.device('cpu:0'):
        with tf.Session() as sess:
            multilayer_hierarchyAdjGen = Multilayer_Hiearchy_adjMatrix_Generator(   sess,
                                                                                    args.Datasets,
                                                                                    args.Desired_graph_size,
                                                                                    args.reconstruction_dir
                                                                                )
            multilayer_hierarchyAdjGen.modelConstruction()

            weight_list, reconstructed_Adj = multilayer_hierarchyAdjGen.train(training_step=args.training_step)
            print("=======================\nreconstuction DOWN!\n=======================")
            if debugFlag is True:
                print('weight list: ', weight_list)
                print('reconstructed_adj shape: ', reconstructed_Adj.shape)

            C.main(filename=args.Datasets[0],type="Hierarchy",constructed_graph=reconstructed_Adj)

            # save Model
            pickle.dump(reconstructed_Adj, open("WS1_WS2_reconstructed_%s.adj",'wb'))






if __name__ == '__main__':
    main(parse_args())
    print('DOWN')