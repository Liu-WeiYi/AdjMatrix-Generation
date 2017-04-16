#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  网络中的每一个小块都会有自己的类别，我们利用这些类别+小块内部的拓扑结构，生成网络---based on Condition GAN
  Created:  04/12/17
  Usage:    python3 Topology_MAIN.py <filename> <MatSize> <classNum>
  Road Map:
  step.0  Edge_list 数据, 分析 节点个数 / 边个数, 并在 对应的文件夹下 生成对应的undirected graph: filename.graph
  step.1  给定每一个小块的大小，并据此划分 filename.metis_graph 网络, 此时会在数据文件所在的文件夹中生成partition文件: filename_MatSize.metis_graph.part.[K]
  step.2  程序主要步骤，主要为了生成
          [Condition GAN Input:] 1. <filename>_MatSize.graph       --- adj字典,     Key: 表示序号(共有nparts个), Value: 为一个 MatSize * MatSize 的矩阵
          [Condition GAN Input:] 2. <filename>_MatSize.degree      --- degree字典,  Key: 表示序号(共有nparts个), Value: 为adj中当前矩阵所属类别, 采用One-Hot Vector表示
          [Condition GAN Input:] 3. <filename>_MatSize.outNeighbor --- 每一个part的出度都连得是网络中的哪些节点
          [Condition GAN Input:] 4. <filename>_MatSize.map         --- partition结果中的每一个Part 是adj中的哪一块(在拼回网络的时候有用)
"""

import os
import glob
import pickle
import sys
import numpy as np
import networkx as nx
import math

debugFlag = False

# ------------------------------
# generate_graph --- step.0
# @input:  当前edge_list所在地址, filename网络文件名, MatSize每一块的大小 (当为-1时只输出网络)
# @output: <filename>.metis_graph --- 用于metis进行划分的网络格式
# @return: 生成的图的路径metis_path, 当前网络总规模graph_size, 当前网络本身(nx对象)graph
# ------------------------------
def generate_graph(path, filename, MatSize=-1):
    edge_list_file = open(glob.glob(path+"*.edge_list")[0], 'r+')

    # step.0
    # 采用nx.graph保存当前图
    g = nx.Graph(name=filename)
    for line in edge_list_file.readlines():
        try:
            src,dst = line.strip().split()
            """
            注: gpmetis只支持节点从1开始!
            """
            src = int(src) + 1
            dst = int(dst) + 1
            g.add_nodes_from([src,dst])
            g.add_edge(src,dst)
        except:
            print('current line is not readable: ', line)
    if debugFlag is True:
        print('File Read Down...')
    edge_list_file.close()

    if MatSize == -1:
        return g
    elif MatSize > 0:
        # 生成网络
        metis_path = path+filename+'_%d.metis_graph'%MatSize
        metis_graph_file = open(metis_path,'w+')
        nodes_number = len(g.nodes())
        edges_number = len(g.edges())
        metis_graph_file.write(' '+str(nodes_number)+' '+str(edges_number)+'\n')

        for node in g.nodes():
            neighbors = nx.neighbors(g,node)
            #  注意，这里应该生成的是directed graph!!!
            for dst_idx in range(len(neighbors)):
                if dst_idx != len(neighbors)-1:
                    metis_graph_file.write(' '+str(neighbors[dst_idx])+'  ')
                else:
                    metis_graph_file.write(' '+str(neighbors[dst_idx])+'\n')
        if debugFlag is True:
            print('metis Write Down...')
        metis_graph_file.close()

        return metis_path, len(g.nodes()), g

# ------------------------------
# metis_graph --- step.1
# @input:  划分出的文件所保存的地址metis_path, 根据需要的网络规模将该网络划分的份数nparts
# @output: metis划分出的网络 <filename>.metis_graph.part.<nparts>
# ------------------------------
def metis_graph(path,nparts):
    os.system('./gpmetis %s %d'%(path,nparts))

# ------------------------------
# generate_AdjMat --- step.2
# @input:  划分出的文件所保存的地址path, 当前网络本身(nx对象)graph, MatSize每一块的大小, 指定的类别个数classNum
# @output: <filename>_MatSize.graph       --- adj字典,     Key: 表示序号(共有nparts个), Value: 为一个 MatSize * MatSize 的矩阵
#          <filename>_MatSize.degree      --- degree字典,  Key: 表示序号(共有nparts个), Value: 为adj中当前矩阵所属类别, 采用One-Hot Vector表示
#          <filename>_MatSize.outNeighbor --- 每一个part的出度都连得是网络中的哪些节点
#          <filename>_MatSize.map         --- partition结果中的每一个Part 是adj中的哪一块(在拼回网络的时候有用)
# @return: maxsize 划分出来的所有块中的最大的大小, (max_size 生成 Adj-Mat, 使得生成的Adj-Mat都是一样大小的!)
# ------------------------------
def generate_AdjMat(path, graph, MatSize, classNum):
    # 1. 获取每一个节点的类别
    node_part = {}
    metis_result_file = open(glob.glob(path+"*_%d.metis_graph.part.*"%MatSize)[0],'r+')
    src = 1
    for line in metis_result_file.readlines():
        part = line.strip()
        if part not in node_part.keys():
            node_part[part] = [src]
        else:
            node_part[part].append(src)
        src += 1
    if debugFlag is True:
        print("Read Metis Part File Down...")

    # 2. 生成Adj-Mat等~
    if debugFlag is True:
        size = {}
        for n in node_part.keys():
            currentLen = len(node_part[n])
            if currentLen not in size.keys():
                size[currentLen] = [n]
            else:
                size[currentLen].append(n)
        print('partition results: ', size.keys())
    ## 获取划分出来块的最大的大小
    max_size = max(set([len(node_part[part]) for part in node_part.keys()]))
    if debugFlag is True:
        print('max part size: ', max_size)
    ## 生成 每一个点对应的邻接矩阵中的坐标-part2Node, 邻接矩阵-adj, 对应的-outDegree, map
    part2Node = {}        # 每一个小块对应的序号(从这个序号可以索引adj,adj_outNeighbor,outDegree相应位置)
    adj = {}              # 每一个小块的邻接矩阵
    adj_outNeighbor = {}  # 每一个小块连接外部的节点的记录
    outDegree = {}        # 每一个小块的类别

    count = 0
    for p in node_part.keys():
        current_part = node_part[p]
        """ extract subgraph :) """
        current_subgraph = graph.subgraph(current_part)
    ### ------------------------------------------------------------
    ### 1. 根据 max_size 生成 part2Node & Adj-Mat, 同时应该保证 生成的Adj-Mat都是一样大小的
        current_part2Node, current_subgraph_adj = graph2Adj(current_subgraph,max_size)

        part2Node[count]    = current_part2Node
        adj[count]          = current_subgraph_adj
    ### ------------------------------------------------------------
    ### 2. 记录adj_outNeighbor --- 采用sum进行
        outNeighbors = {}
        for node in current_part:
            neighbors = nx.neighbors(graph, node)
            outNeighbors[node] = list(set(neighbors)-set(current_part))

        adj_outNeighbor[count] = outNeighbors
        ### ------------------------------------------------------------
        count += 1

    ### ------------------------------------------------------------
    ### 3. 根据adj_outNeighbor信息判断每一小块属于的类别
    outDegreeLength = sorted([sum([len(adj_outNeighbor[part][node]) for node in adj_outNeighbor[part].keys()]) for part in adj_outNeighbor.keys()])
    if classNum > 1:
        step = (outDegreeLength[-1]-outDegreeLength[0])//(classNum-1)
    elif classNum == 1:
        step = np.inf
    if debugFlag is True:
        check = set()
    for part in adj_outNeighbor.keys(): # 保证此时每一个part 就是上面的 count 信息不变
        outDegreeVector = np.zeros(shape=[classNum])
        current_part_length = sum([len(adj_outNeighbor[part][node]) for node in adj_outNeighbor[part].keys()])
        group = math.floor(current_part_length/step)
        if debugFlag is True:
            check.add(group)
        # 找出类别之后，在相应位置变换成1
        if classNum > 0:
            outDegreeVector[group] = 1
        # 存之~
        outDegree[part] = outDegreeVector
    if debugFlag is True:
        original_part = set([i for i in range(classNum)])
        print('missing group: ', original_part-check)

    # 3. 写文件~
    pickle.dump(part2Node, open(path+graph.name+'_%d.map'%max_size,'wb'))
    pickle.dump(adj, open(path+graph.name+'_%d.graph'%max_size,'wb'))
    pickle.dump(outDegree, open(path+graph.name+'_%d.degree'%max_size,'wb'))
    pickle.dump(adj_outNeighbor, open(path+graph.name+'_%d.outNeighbor'%max_size,'wb'))

    if debugFlag is True:
        print('\n\nTopology Processing Down...')
        print('[*]adj shape (key * shape): ', len(adj.keys()), adj[0].shape)
        print('[*]degree shape (key * length): ', len(outDegree.keys()), outDegree[0])

    return max_size

# ====================================================================================
# utils
# ====================================================================================
def graph2Adj(g, max_size = 1):
    # 当max_size == -1时，只返回 adj，而不进行padding
    src_dict = {}
    adjIdx_2_node = {}
    adjIdx = 0
    # create adj file
    for src in g.nodes():
        src_dict[src] = []
        adjIdx_2_node[adjIdx] = src
        adjIdx += 1
        for dst in g.nodes():
            if src == dst:
                src_dict[src].append(0)
            else:
                if dst in g.neighbors(src):
                    src_dict[src].append(1)
                else:
                    src_dict[src].append(0)
    adj = np.array([np.array(src_dict[src]) for src in src_dict.keys()])
    if max_size == -1:
        return adj
    elif max_size >= 0:
        # padding adj file to max_size
        padded_adj = np.zeros(shape=[max_size,max_size])
        padded_adj[:adj.shape[0], :adj.shape[1]] = adj

        if debugFlag is True:
            if adj.shape != padded_adj.shape:
                print('original adj shape: ', adj.shape, '\tpadded adj shape: ',padded_adj.shape)
                print(adj, '\n', padded_adj)

        return adjIdx_2_node, padded_adj




# ====================================================================================
# main
# ====================================================================================
if __name__ == "__main__":
    try:
        filename  = sys.argv[1]       # 网络名称
        MatSize   = int(sys.argv[2])  # 分析的矩阵大小(即输入GAN的inputMat大小)
        classNum  = int(sys.argv[3])  # 类别标签
        if classNum == 0:
            print('[!!!] classNum should bigger than 0. As it does not make sense that there is ZERO classes!')
            sys.exit()
    except :
        sys.exit("Usage: python3 Topology_MAIN.py <filename> <MatSize> <classNum>\
                        \n\t <filename> : current Edge List File \
                        \n\t <MatSize>  : Expected Matrix Size for single block \
                        \n\t <classNum> : Expected Classes. classNum should always bigger than 1!!")

    # step.0
    path = os.path.join('data',filename, '')
    metis_path,graph_size, graph = generate_graph(path, filename, MatSize)

    # step.1
    nparts = math.ceil(graph_size/MatSize)
    metis_graph(metis_path, nparts)

    # step.2
    generate_AdjMat(path, graph, MatSize, classNum)