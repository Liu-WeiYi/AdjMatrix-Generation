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
'''
def draw_degree(G):
    degree = nx.degree_histogram(G)
    x = range(len(degree))
    y = [z/float(sum(degree)) for z in degree]

    plt.loglog(x,y,color='blue',linewidth=2)
'''
def draw_degree(reG,oriG,path,figure_name):
    if not os.path.exists(path):
        print("some error happend?")
        os.makedirs(path)

    # 1. draw reconstruct graph~
    plt.figure("reconstruct graph degree distribution")
    degree_reG = nx.degree_histogram(reG)
    x_reG = range(len(degree_reG))
    y_reG = [z/float(sum(degree_reG)) for z in degree_reG]

    plt.loglog(x_reG,y_reG,color='blue',linewidth=2)
    plt.savefig(path+figure_name+".png")
    plt.savefig(path+figure_name+".pdf")

    plt.clf()

    # 2. draw reconstruct graph and original graph on a graph~
    plt.figure("reGraph and originGraph")
    degree_oriG = nx.degree_histogram(oriG)
    x_oriG = range(len(degree_oriG))
    y_oriG = [z/float(sum(degree_oriG)) for z in degree_oriG]

    plt.loglog(x_reG,y_reG,color='blue',linewidth=2)
    plt.loglog(x_oriG,y_oriG,color='red',linewidth=2)

    plt.savefig(path+"combined_"+figure_name+".png")
    plt.savefig(path+"combined_"+figure_name+".pdf")

    plt.clf()

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
    '''
    plt.figure("original graph degree distribution")
    draw_degree(original_graph)
    print('original edge number: ',len(original_graph.edges()))
    '''

    # 2. reconstruct graph
    if constructed_graph == -1:
        reconstruct_graph_path = os.path.join("reconstruction", filename, type,"")
        reconstruct_graph_adj = pickle.load(open(glob.glob(reconstruct_graph_path+"*.adj")[0],'rb'))
    else:
        reconstruct_graph_adj = constructed_graph
    reconstruct_graph = adj2Graph(reconstruct_graph_adj, edgesNumber = len(original_graph.edges()))
    print('edge number: ', len(reconstruct_graph.edges()))
    #plt.figure("reconstruct graph degree distribution")
    draw_degree(reconstruct_graph, original_graph, os.path.join("reconstruction", filename, type, ""), filename)

    print("Clustering: ",nx.average_clustering(original_graph), ' ', nx.average_clustering(reconstruct_graph))
    # print("Diameter: ", nx.average_shortest_path_length(original_graph), ' ', nx.average_shortest_path_length(reconstruct_graph))
    # print("degree centrality: ", nx.degree_centrality(original_graph), ' ',  nx.degree_centrality(reconstruct_graph))
    #print("closeness centrality: ", nx.closeness_centrality(original_graph), ' ', nx.closeness_centrality(reconstruct_graph))



if __name__ == "__main__":
    try:
        filename  = sys.argv[1]     # 网络名称
        type = sys.argv[2]     # 构建网络的方式
    except :
        sys.exit("Usage: python3 Compair.py <filename> <Type>\
                        \n\t <filename> : current Edge List File\
                        \n\t <Type>     : Reconstruction Type [Hierarchy]")

    main(filename, type)
