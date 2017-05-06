#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  1. Calculate Degree Distribution
            2. Calculate Clustering Coefficient
            3. Calculate Diamter
            4. Calculate Node Centrality
  Created: 05/05/17
"""
import pickle
import matplotlib.pyplot as plt
import sys
import os
import glob
from Compair import adj2Graph
import networkx as nx


net_name = sys.argv[1]

# 读入原网络
[trained_graph_adj_list,origin_adj] = pickle.load(open('%s_adjs.pickle'%net_name,'rb'))
oriG = adj2Graph(net_name,origin_adj,1000,0.5)
# 规定Cut
# BA
# cut_value_list = ['0.6224','0.6136','0.6117','0.6106','0.6063','0.6028']

# WS
# cut_value_list = ["0.608868","0.605854","0.599059","0.596019","0.595348","0.591385"]

# ER
# cut_value_list = ["0.622459","0.60469","0.511051","0.502593","0.500775"]

# Kron
# cut_value_list = ["0.622459","0.597827","0.581947","0.571054","0.559493","0.556555","0.535661","0.519301","0.493415"]

# Facebook
# cut_value_list = ["0.622459","0.616346","0.572501","0.567079","0.566142","0.560701"]

# Wiki-Vote
cut_value_list = ["0.622459","0.618375","0.609502"]


# 提取相应的re_adj
adj_path = os.path.join("adjs",net_name,"")
graph_list = [oriG]
all_file_path = glob.glob(adj_path+"%s_adjs-RE-*.pickle"%(net_name))
for cut in cut_value_list:
    for file in all_file_path:
        if cut in file:
            current_file_path = file
            print('current file path: ', current_file_path)
            current_adj_RE = pickle.load(open(current_file_path,'rb'))
            current_reG = adj2Graph(net_name, current_adj_RE, 1000,0.5)
            graph_list.append(current_reG)
            break

# 计算并保存相关
# 注意第一个是原图哈！！
features = {'fq':[],'cc':[],'d':[],'nc':[]}
for g in graph_list:
    # fq
    degrees = nx.degree_histogram(g)
    frequency = [z/float(sum(degrees)) for z in degrees]
    features['fq'].append(frequency)
    # cc
    avg_cc = nx.average_clustering(g)
    features['cc'].append(avg_cc)

    # dc
    avg_dc = sum(nx.degree_centrality(g))/(len(g.nodes()))
    features['nc'].append(avg_dc)

    # d
    if not nx.is_connected(g):
        da = nx.diameter(max(nx.connected_component_subgraphs(g), key=len))
    else:
        da = nx.diameter(g)
    features['d'].append(da)

# 打印到控制台~
for i in features['cc']:
    print(i)
print('\n#############################################\n')
for i in features['nc']:
    print(i)
print('\n#############################################\n')
for i in features['d']:
    print(i)
print('\n#############################################\n')
for degree in features['fq']:
    for i in degree:
        print(i)
    print('##################')
print('\n#############################################\n')






