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
import pandas as pd
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
def draw_degree(cutValue, reG,oriG,path,net_name, num):

    figure_name = net_name

    # pos = nx.spring_layout(oriG)

    # # ori
    # plt.figure('ori Fig')
    # nx.draw_networkx_nodes(oriG,pos,node_color='w')
    # nx.draw_networkx_edges(oriG,pos,edge_color='k',alpha=0.8)
    # nx.draw_networkx_labels(oriG,pos)

    # # reG
    # plt.figure('reG Fig')
    # pos = nx.spring_layout(reG)
    # nx.draw_networkx_nodes(reG,pos,node_color='w')
    # nx.draw_networkx_edges(reG,pos,edge_color='k',alpha=0.8)
    # nx.draw_networkx_labels(reG,pos)


    # oriAdj
    [trained_graph_adj_list,origin_adj] = pickle.load(open('%s_adjs.pickle'%net_name,'rb'))
    # reAdj
    reAdj = graph2Adj(reG,-1)

    # 存储所有Cut下的Adj
    path = os.path.join("adjs",net_name,"")
    if not os.path.exists(path):
        os.makedirs(path)
    pickle.dump(reAdj, open(path+'%s_adjs-RE-%f.pickle'%(net_name,cutValue),'wb'))

    # 比较与原始Adj的距离
    A = origin_adj-reAdj
    L1_norm = np.linalg.norm(np.sum(A))
    L2_norm = np.linalg.norm(A)
    F_norm  = np.linalg.norm(A,ord='fro')

    norm_file = open(path+"%s_norm.csv"%net_name,'a')
    print('L1_norm: ',L1_norm, 'L2_norm: ',L2_norm, 'F_norm: ', F_norm)
    norm_file.write("%f,%f,%f\n"%(cutValue, L1_norm, L2_norm))
    norm_file.close()
    os.system('python3 draw_subgraph.py %s %f %s'%(net_name,cutValue,path))




    # 画adj图...
    plt.figure('oriAdj...')
    plt.imshow(origin_adj,cmap='Purples',interpolation='none')
    plt.colorbar()
    plt.figure('reAdj...')
    plt.imshow(reAdj,cmap='Purples',interpolation='none')
    plt.colorbar()

    # plt.show()



    # # plt.show()

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

    # plt.clf()

    # 2. draw reconstruct graph and original graph on a graph~
    plt.figure("reGraph and originGraph")
    degree_oriG = nx.degree_histogram(oriG)
    x_oriG = range(len(degree_oriG))
    y_oriG = [z/float(sum(degree_oriG)) for z in degree_oriG]

    plt.loglog(x_reG,y_reG,color='blue',linewidth=2)
    plt.loglog(x_oriG,y_oriG,color='red',linewidth=2)

    plt.savefig(path+"combined_"+figure_name+".png")
    plt.savefig(path+"combined_"+figure_name+".pdf")

    # re_df = pd.DataFrame({'reconstruction': y_reG})
    # ori_df = pd.DataFrame({'origin': y_oriG})
    # out_df = pd.concat([re_df, ori_df], ignore_index=True, axis=1)
    # out_df.to_csv(path+"reconstruct_dataset"+figure_name+"%d.csv"%num, sep = '\t', encoding = 'utf-8')

    # plt.clf()
    # plt.show()

def adj2Graph(net_name, adj, edgesNumber=1000, cutvalue=1):
    graph = nx.Graph()

    # 将有向网络变成无向网络
    adj = (adj+np.transpose(adj))/2


    meanValue = np.mean(adj)
    print("mean value: ", meanValue)

    flatten_value = sorted(list(adj.flatten()),reverse=True)
    # 去除0元素
    #flatten_value = [i for i in flatten_value if i != 0]
    #cutvalue = np.mean(np.array(flatten_value))

    # cutvalue = flatten_value[edgesNumber-1]

    flatten_value = list(set(flatten_value))
    print('flatten_value length: ',len(flatten_value))

    if not os.path.exists('%s_cut_value_list.pickle'%net_name):
        pickle.dump(flatten_value, open('%s_cut_value_list.pickle'%net_name,'wb'))
        raise SystemExit('已创建cut_value, 请重新运行程序~~')



    print("cut value: ", cutvalue)

    print(adj.shape)
    for src in range(len(list(adj))):
        if src % 500 == 0:
            print('processed %d nodes'%src)
        graph.add_node(src)
        for dst in range(src, len(adj[src])):
            if src != dst:
                if adj[src][dst] >= cutvalue:
                    # graph.add_edge(src,dst,weight=adj[src][dst])
                    if (dst,src) not in graph.edges():
                        graph.add_edge(src,dst)

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
    draw_degree(reconstruct_graph, original_graph, os.path.join("reconstruction", filename, type, ""), filename, 0)

    print("Clustering: ",nx.average_clustering(original_graph), ' ', nx.average_clustering(reconstruct_graph))
    # print("Diameter: ", nx.average_shortest_path_length(original_graph), ' ', nx.average_shortest_path_length(reconstruct_graph))
    # print("degree centrality: ", nx.degree_centrality(original_graph), ' ',  nx.degree_centrality(reconstruct_graph))
    #print("closeness centrality: ", nx.closeness_centrality(original_graph), ' ', nx.closeness_centrality(reconstruct_graph))



# if __name__ == "__main__":
#     try:
#         filename  = sys.argv[1]     # 网络名称
#         type = sys.argv[2]     # 构建网络的方式
#     except :
#         sys.exit("Usage: python3 Compair.py <filename> <Type>\
#                         \n\t <filename> : current Edge List File\
#                         \n\t <Type>     : Reconstruction Type [Hierarchy]")

#     main(filename, type)

