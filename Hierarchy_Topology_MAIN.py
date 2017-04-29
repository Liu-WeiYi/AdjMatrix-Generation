#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  将网络进行层次化的分割, 对每一个层次单独采用GAN学习该层中的拓扑结构, 之后将这些层进行结合。
            因为不同层下的图分割会 ``切断'' 不一样的连边，这样相当于 利用``层次化''的思想保证了网络信息的不丢失
            从而解决Conditional——Topology_MAIN.py中无法解决的两个块之间无法``自动''生成块间连接的缺陷。
            其中:
            1. 网络 层次化 获得方式  : 原网络的社团结构上 采用 层次化的 快速社团划分算法 可以轻松获得网络的层次结构 😀
            2. 网络 层次化 好处     : 每一层的社团个数 相当于 就是我们 需要进行图分割的 标准。
            3. 这样做的好处在于, 我们可以通过GAN来学习 不同层次下的网络结构。
  Road Map:
  step.0 读入 Edge_list 数据, 采用Louvain算法获取该网络的层次结构 [ Louvain 获取层次结构的工具在 external_tools 中]
  step.1 利用 Conditional_Topology_Main.py 生成每一个层次中对应的:
        [Hierarchy GAN Input:] 1. <filename>_MatSize.graph       --- adj字典,     Key: 表示序号(共有nparts个), Value: 为一个 MatSize * MatSize 的矩阵
        [Hierarchy GAN Input:] 2. <filename>_MatSize.degree      --- degree字典,  Key: 表示序号(共有nparts个), Value: 为adj中当前矩阵所属类别, 采用One-Hot Vector表示
        [X]                    3. <filename>_MatSize.outNeighbor --- 每一个part的出度都连得是网络中的哪些节点 [ 在这里不需要~ ]>
        [Hierarchy GAN Input:] 4. <filename>_MatSize.map         --- partition结果中的每一个Part 是adj中的哪一块(在拼回网络的时候有用)
  Created: 04/13/17
"""
import time
import os
import sys
import glob
import math
import pickle
import networkx as nx

import Conditional_Topology_MAIN as CTM
from external_tools.community_louvain import generate_dendrogram

debugFlag = False


if __name__ == "__main__":
    try:
        filename  = sys.argv[1]       # 网络名称
    except :
        sys.exit("Usage: python3 Topology_MAIN.py <filename> \
                        \n\t <filename> : current Edge List File")

    # step.0 获取层次~
    if debugFlag is True:
        start = time.time()
        print('get layers info...', end='\t')
    path = os.path.join('data',filename, '')
    graph = CTM.generate_graph(path, filename, -1)
    dendrogram = generate_dendrogram(graph)
    layer_size = []
    for layer in dendrogram: # 获取每一层的社团划分结果
        com = set()
        for node in layer.keys(): # 对每一层，记录社团个数
            com.add(layer[node])
        layer_size.append(len(com))
    layer_size = layer_size[:4]
    if debugFlag is True:
        print('used time: %.4f\tnet-%s layers infor\t'%(time.time()-start, filename), len(layer_size), layer_size)

    partition_info = []
    for nparts in layer_size:
        """inherit from Conditional_Topology_MAIN.py"""
        """layer的作用是指导图分割每层分割出的社团个数, 而非 每张子图大小, 所以在这里反而应该给出每张子图的大小"""
        MatSize = math.ceil(len(graph.nodes())/nparts)

        # step.0
        path = os.path.join('data',filename, '')
        metis_path, graph_size, graph = CTM.generate_graph(path, filename, MatSize)

        # step.1
        CTM.metis_graph(metis_path, nparts)

        # step.2
        ## 由于不需要事先指定类别，所以这里类别 classNum 直接等于1
        current_max_size = CTM.generate_AdjMat(path, graph, MatSize, classNum=1)
        partition_info.append([nparts,current_max_size])

        # os.system('python3 Conditional_Topology_MAIN.py %s %d 1'%(filename, npart))
        # 不需要 .metis_graph 和 .outdegree 文件 故删除以节省空间~
        os.system('rm %s*.metis_graph'%path)
        os.system('rm %s*.outNeighbor'%path)

    pickle.dump(partition_info, open('%s_partition_info.pickle'%filename,'wb'))
    if debugFlag is True:
        print('partion info--[trainable_data_size, inputMatSize]\n\t\t', partition_info)
