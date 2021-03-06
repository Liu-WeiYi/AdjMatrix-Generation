#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Compair Reconstruction Results~~
  Created:  04/15/17
"""
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import os
import glob
import sys
import numpy as np
import math

from Conditional_Topology_MAIN import graph2Adj, generate_graph

def draw_degree(G):
    degree = nx.degree_histogram(G)
    x = range(len(degree))
    y = [z/float(sum(degree)) for z in degree]

    plt.loglog(x,y,color='blue',linewidth=2)

def adj2Graph(adj, edgesNumber):
    graph = nx.Graph()

    # 将有向网络变成无向网络
    adj = (adj+np.transpose(adj))/2


    meanValue = np.mean(adj)
    print("mean value: ", meanValue)

    flatten_value = sorted(list(adj.flatten()),reverse=True)
    # 去除0元素
    #flatten_value = [i for i in flatten_value if i != 0]
    #cutvalue = np.mean(np.array(flatten_value))
    cutvalue = flatten_value[edgesNumber-1]

    print("cut value: ", cutvalue)

    for src in range(len(list(adj))):
        graph.add_node(src)
        for dst in range(len(adj[src])):
            if src != dst:
                if adj[src][dst] >= cutvalue:
                    graph.add_edge(src,dst,weight=adj[src][dst])

    return graph


def main(filename, type, constructed_graph = -1):
    # 1. original graph
    original_graph_path = os.path.join("data",filename,"")
    original_graph = generate_graph(original_graph_path,filename,-1)
    plt.figure("original graph degree distribution")
    draw_degree(original_graph)
    print('original edge number: ',len(original_graph.edges()))


    # 2. reconstruct graph
    # if type(constructed_graph) is int:
    if 1+1 == 1:
        reconstruct_graph_path = os.path.join("reconstruction", filename, type,"")
        reconstruct_graph_adj = pickle.load(open(glob.glob(reconstruct_graph_path+"*.adj")[0],'rb'))
    else:
        reconstruct_graph_adj = constructed_graph
    reconstruct_graph = adj2Graph(reconstruct_graph_adj, edgesNumber = len(original_graph.edges()))
    print('edge number: ', len(reconstruct_graph.edges()))
    plt.figure("reconstruct graph degree distribution")
    draw_degree(reconstruct_graph)

    print("Clustering: ",nx.average_clustering(original_graph), ' ', nx.average_clustering(reconstruct_graph))
    # print("Diameter: ", nx.average_shortest_path_length(original_graph), ' ', nx.average_shortest_path_length(reconstruct_graph))
    # print("degree centrality: ", nx.degree_centrality(original_graph), ' ',  nx.degree_centrality(reconstruct_graph))
    #print("closeness centrality: ", nx.closeness_centrality(original_graph), ' ', nx.closeness_centrality(reconstruct_graph))

    plt.show()




if __name__ == "__main__":
    try:
        filename  = sys.argv[1]     # 网络名称
        type = sys.argv[2]     # 构建网络的方式
    except :
        sys.exit("Usage: python3 Compair.py <filename> <Type>\
                        \n\t <filename> : current Edge List File\
                        \n\t <Type>     : Reconstruction Type [Hierarchy]")

    main(filename, type)