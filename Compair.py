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

def draw_degree(reG,oriG,path,figure_name):
    if not os.path.exists(path):
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

# ------------------------------
# compair_main()
# @purpose: 把重构的邻接矩阵和真实的邻接矩阵进行比较~ 返回比较结果
# @input:   filename 当前网络名称, type 生成网络的类型，这里默认为Hierarchy
#           adjDisk=True 为TRUE的话读取磁盘上的reconstructed_graph.adj文件，如果不是，则这里直接传入为TRUE的话读取reconstructed_graph
#           permutation_step=0 指定permutate次数 step=100 指定步长
# @output: 在 reconstruction/<type>/<filename>/
#          1. 度分布的比较图 --- a). filename_permutation_step.png[.pdf] b). combine_filename_permutation_step.png[.pdf]
#          2. filename+"_"+str(permutation_step)+"_"+str(step)_clustering_coefficient.txt
# ------------------------------
def compair_main(filename, type, adjDisk=True, permutation_step=0, step=100):
    # 1. original graph
    original_graph_path = os.path.join("data",filename,"")
    original_graph = generate_graph(original_graph_path,filename,-1)

    # plt.figure("original graph degree distribution")
    # draw_degree(original_graph)
    print('original edge number: ',len(original_graph.edges()))

    # 2. reconstruct graph
    if adjDisk is True:
        reconstruct_graph_path = os.path.join("reconstruction", filename, type,"")
        reconstruct_graph_adj = pickle.load(open(glob.glob(reconstruct_graph_path+"*_%s.adj"%filename)[0],'rb'))
    else:
        reconstruct_graph_adj = adjDisk

    reconstruct_graph = adj2Graph(reconstruct_graph_adj, edgesNumber = len(original_graph.edges()))
    print('edge number: ', len(reconstruct_graph.edges()))

    path        = os.path.join("reconstruction",filename, type, "")
    figure_name = filename+"_"+str(permutation_step)+"_"+str(step)
    draw_degree(reconstruct_graph, original_graph, path, figure_name)

    oriG_CC = nx.average_clustering(original_graph)
    reG_CC = nx.average_clustering(reconstruct_graph)
    print("Clustering: ", oriG_CC, ' ', reG_CC)
    # print("Diameter: ", nx.average_shortest_path_length(original_graph), ' ', nx.average_shortest_path_length(reconstruct_graph))
    # print("degree centrality: ", nx.degree_centrality(original_graph), ' ',  nx.degree_centrality(reconstruct_graph))
    #print("closeness centrality: ", nx.closeness_centrality(original_graph), ' ', nx.closeness_centrality(reconstruct_graph))

    cc_file = open(path+figure_name+"_clustering_coefficient.txt",'w+')
    cc_file.write(str(oriG_CC)+"\t"+str(reG_CC)+"\n")


if __name__ == "__main__":
    try:
        filename  = sys.argv[1]     # 网络名称
        type = sys.argv[2]     # 构建网络的方式
    except :
        sys.exit("Usage: python3 Compair.py <filename> <Type>\
                        \n\t <filename> : current Edge List File\
                        \n\t <Type>     : Reconstruction Type [Hierarchy]")

    compair_main(filename, type)