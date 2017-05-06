#!/usr/bin/env python
#coding:utf-8
"""
  Author:   weiyiliu --<weiyiliu@us.ibm.com>
  Purpose:  Calculate NMI after Community Detection
  Created: 05/05/17
"""

import os, glob, sys
import pickle
from external_tools.community_louvain import best_partition
from external_tools.NMI_Calculation import nmi_non_olp
from Compair import adj2Graph
from sklearn.metrics.cluster import normalized_mutual_info_score
import math
import random
import networkx as nx


def Louvain(G):
    coms = []
    partition = best_partition(G)
    for com in set(partition.values()):
        list_nodes = [nodes for nodes in partition.keys() if partition[nodes] == com]
        coms.append(list_nodes)
    return coms

#----------------------------------------------------------------------
def NMI(self,com_type1,com_type2):
    """
    com_type1, com_type2 --- two communities
    """
    nmi = mni_olp_1(com_type1, com_type2)
    if nmi - 0.5 < 0.00001:
        print('NMI = 0.5 Community detection failed')
    return nmi


def ourMethod(net_name):
    """
    @purpose: return original graph and sampled graphs list
    """
    # Read Original Graph
    [trained_graph_adj_list,origin_adj] = pickle.load(open('%s_adjs.pickle'%net_name,'rb'))
    oriG = adj2Graph(net_name,origin_adj,1000,0.5)

    if net_name == "BA":
        cut_value_list = ['0.6224','0.6136','0.6117','0.6106','0.6063','0.6028']
    elif net_name == "WS":
        cut_value_list = ["0.608868","0.605854","0.599059","0.596019","0.595348","0.591385"]
    elif net_name == "ER":
        cut_value_list = ["0.622459","0.60469","0.511051","0.502593","0.500775"]
    elif net_name == "kron":
        cut_value_list = ["0.622459","0.597827","0.581947","0.571054","0.559493","0.556555","0.535661","0.519301","0.493415"]
    elif net_name == "facebook":
        cut_value_list = ["0.622459","0.616346","0.572501","0.567079","0.566142","0.560701"]
    elif net_name == "wiki-vote":
        cut_value_list = ["0.622459","0.618375","0.609502"]

    # 提取相应的re_adj
    adj_path = os.path.join("adjs",net_name,"")
    graph_list = []
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

    return oriG, graph_list

def randomDelete(oriG, net_name):
    """
    @purpose: create reG based on randomly delete edges in original graph
    """
    reG_list = []
    if net_name == "BA":
        delete_sample = ["0.805220884","0.736947791","0.639558233","0.606425703","0.584337349","0.425702811"]
    elif net_name == "WS":
        delete_sample = ["0.888","0.886","0.84","0.82","0.454","0.022"]
    elif net_name == "ER":
        delete_sample = ["0.956778074","0.782735131","0.05091025"]
    elif net_name == "kron":
        delete_sample = ["0.12229614","0.113452575","0.082420428","0.081066008","0.075329642","0.066804764","0.039397682","0.029478548","0.014261244"]
    elif net_name == "facebook":
        delete_sample = ["0.477219666","0.166681778","0.125053834","0.085907927","0.096924088","0.080513181"]
    elif net_name == "wiki-vote":
        delete_sample = ["0.416910183","0.262072158","0.144007561"]

    total_edges = len(oriG.edges())
    for per in delete_sample:
        newG = nx.Graph()
        newG.add_nodes_from(oriG.nodes())

        remain_edges = math.floor(float(per)*total_edges)
        edges = random.sample(oriG.edges(),remain_edges)
        newG.add_edges_from(edges)

        reG_list.append(newG)
    return reG_list


def clean(coms, isolated_nodes):
    """
    delete node in coms according to isolated_nodes
    """
    new_coms = []
    for com in coms:
        clean_com = [node for node in com if node not in isolated_nodes]
        new_coms.append(clean_com)
    return new_coms

if __name__ == "__main__":
    """Now We Have: ["BA","kron","WS","ER","wiki-vote","facebook"]"""
    # for dirc in ["BA","kron","WS","ER","facebook","wiki-vote"]:
    net_name = sys.argv[1]

    """Now we prepare the original graph and sampled graphs (stored in a list)"""
    # Our Method
    oriG, reG_list = ourMethod(net_name)
    # sampling Value used for Other methods!!
    """other methods in here"""
    # for example: Randomly deleting edges
    reG_list = randomDelete(oriG, net_name)

    """Find Communities and calculate NMI"""
    # find com for original graph
    oriG_com = Louvain(oriG)
    for reG in reG_list:
        reG_com = Louvain(reG)
        reG_isolated_nodes = [com[0] for com in reG_com if len(com)==1]
        reG_com = [com for com in reG_com if len(com)>2]
        # delete isolated nodes~
        oriG_com = clean(oriG_com,reG_isolated_nodes)
        oriG_com = [com for com in oriG_com if len(com)>2]

        nmi = nmi_non_olp(oriG_com,reG_com)
        print(nmi)





