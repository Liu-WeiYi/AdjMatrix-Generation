#coding: utf-8
import networkx as nx
node_size = 28
graph_num = 20000

import numpy as np
import pickle
import tensorflow as tf
import os
import glob
import matplotlib.pyplot as plt

adj = {}
degree = {}

for i in range(graph_num):
    if i % 100 == 0:
        print('generate %d graph...'%i)
    g = nx.barabasi_albert_graph(node_size,2)
    # current_file = open(str(i)+'.adj','w+')
    src_dict = {}
    # create adj file
    for src in g.nodes():
        src_dict[src] = []
        for dst in g.nodes():
            if src == dst:
                src_dict[src].append(0)
            else:
                if dst in g.neighbors(src):
                    src_dict[src].append(1)
                else:
                    src_dict[src].append(0)

    # store in numpy
    adj[i] = np.array([np.array(src_dict[src]) for src in src_dict.keys()])
    degree[i] = [0 for _ in range(node_size)]

    # # write into file
    # for src in src_dict.keys():
    #     for idx in range(len(src_dict[src])):
    #         if idx != len(src_dict[src])-1:
    #             current_file.write(str(src_dict[src][idx])+' ')
    #         else:
    #             if src != list(src_dict.keys())[-1]:
    #                 current_file.write(str(src_dict[src][idx])+'\n')
    #             else:
    #                 current_file.write(str(src_dict[src][idx]))
    # # write degree file
    # current_degree_file = open(str(i)+'.degree','w+')
    # for n_idx in range(node_size):
    #     if n_idx != node_size-1:
    #         current_degree_file.write(str(0)+'\n')
    #     else:
    #         current_degree_file.write(str(0))
    # # close file
    # current_file.close()
    # current_degree_file.close()

pickle.dump(adj, open('%d_size_%d_WS.graph'%(node_size,graph_num),'wb'))
pickle.dump(degree, open('%d_size_%d_WS.degree'%(node_size,graph_num),'wb'))
# # 将adj字典转换成Tensor
# for i in adj.keys():
#     if i == 0:
#         Tensor = tf.stack([adj[i]],axis=0)
#     else:
#         slice = tf.stack([adj[i]],axis=0)
#         Tensor = tf.concat([Tensor, slice],axis=0)

# Tensor = tf.expand_dims(input=Tensor, axis=-1)
# print(Tensor.shape)
print('DOWN')



def save_topology(adj, sample_folder, dataset_name, graph_name):
    graph = nx.Graph()
    path = sample_folder+'/'+dataset_name

    if not os.path.isdir(path):
        os.makedirs(path)
    # 1. transfer adj to nx
    # adj_list = list(np.squeeze(adj[0,:,:,:]))
    adj_list = list(adj)

    for src in range(len(adj_list)):
        graph.add_node(src)
        for dst in range(len(adj_list[src])):
            if adj_list[src][dst] >= 0.2: # 防止 sample 出现 不在 [0,1]的情况
                graph.add_edge(src,dst)

    # 2. read position
    pos_file = glob.glob(path+'/*.pos')
    if pos_file == []:
        pos = nx.spring_layout(graph)
        pickle.dump(pos, open(path+'/graph.pos','wb'))
    else:
        pos = pickle.load(open(pos_file[0],'rb'))

    # 3. draw graph
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color='b', alpha=0.8)
    nx.draw_networkx_edges(graph, pos, width=1.5, alpha=0.8)
    nx.draw_networkx_labels(graph, pos, font_color='w')

    plt.savefig(path+'/'+graph_name+'.png')
    plt.savefig(path+'/'+graph_name+'.pdf')
    # plt.show()

    # 4. store graph
    pickle.dump(graph, open(path+'/'+graph_name+'.graph','wb'))


save_topology(adj=adj[0], sample_folder="./samples", dataset_name="WS_test", graph_name="train_0_1")