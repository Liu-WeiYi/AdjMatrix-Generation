#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  cut the graph into hierarchical layers:
        [Hierarchy GAN Input:] 1. <filename>_MatSize.graph       --- dic for adj,     Key: current part(nparts), Value: MatSize * MatSize matrix
        [Hierarchy GAN Input:] 2. <filename>_MatSize.degree      --- dic for degree,  Key: current part(nparts), Value: One-Hot Vector
        [X]                    3. <filename>_MatSize.outNeighbor --- for CONDITION GAN --- No use in here
        [Hierarchy GAN Input:] 4. <filename>_MatSize.map         --- partition results. To indicate current part is locate in where in adj
  Created: 04/13/17
"""
import time
import os
import sys
import glob
import math
import pickle

import Conditional_Topology_MAIN as CTM
from external_tools.community_louvain import generate_dendrogram

debugFlag = True


if __name__ == "__main__":
    try:
        filename  = sys.argv[1]       # network datasets
    except :
        sys.exit("Usage: python3 Topology_MAIN.py <filename> \
                        \n\t <filename> : current Edge List File")

    # step.0 get layers
    if debugFlag is True:
        start = time.time()
        print('get layers info...', end='\t')
    path = os.path.join('data',filename, '')
    graph = CTM.generate_graph(path, filename, -1)
    dendrogram = generate_dendrogram(graph)
    layer_size = []
    for layer in dendrogram: # get each layer community numbers
        com = set()
        for node in layer.keys():
            com.add(layer[node])
        layer_size.append(len(com))
    if debugFlag is True:
        print('used time: %.4f\tnet-%s layers infor\t'%(time.time()-start, filename), len(layer_size), layer_size)

    partition_info = []
    for nparts in layer_size:
        """inherit from Conditional_Topology_MAIN.py"""
        MatSize = math.ceil(len(graph.nodes())/nparts)

        # step.0
        path = os.path.join('data',filename, '')
        metis_path, graph_size, graph = CTM.generate_graph(path, filename, MatSize)

        # step.1
        CTM.metis_graph(metis_path, nparts)

        # step.2
        current_max_size = CTM.generate_AdjMat(path, graph, MatSize, classNum=1)
        partition_info.append([nparts,current_max_size])

        # os.system('python3 Conditional_Topology_MAIN.py %s %d 1'%(filename, npart))
        # remove  .metis_graph & .outdegree just to save space :)
        os.system('rm %s*.metis_graph'%path)
        os.system('rm %s*.outNeighbor'%path)

    pickle.dump(partition_info, open('%s_partition_info.pickle'%filename,'wb'))
    if debugFlag is True:
        print('partion info--[trainable_data_size, inputMatSize]\n\t\t', partition_info)