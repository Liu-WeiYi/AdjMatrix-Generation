# draw subgraphs and sub adjs

import networkx as nx
import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import os

def adj2graph(adj,beginPoint=0):
    g = nx.Graph()
    src = 0+beginPoint
    for dst_list in adj:
        g.add_node(src)
        for dst in range(len(dst_list)):
            if dst_list[dst] == 1:
                g.add_edge(src,dst+beginPoint)
        src += 1
    return g

# graph_name = "wiki-vote"
# graph_name = "facebook"
graph_name = sys.argv[1]
cutValue = sys.argv[2]
path = sys.argv[3]



# 1. oriAdj
[trained_graph_adj_list,origin_adj] = pickle.load(open('%s_adjs.pickle'%graph_name,'rb'))
# 2. reAdj
reAdj = pickle.load(open(path+'%s_adjs-RE-%s.pickle'%(graph_name,cutValue),'rb'))
# reshape...
flag = True
beginPoint = 0
step = 20

endPoint = beginPoint+step
print('begin point: ', beginPoint, "end point: ", endPoint)

origin_adj = origin_adj[beginPoint:endPoint,beginPoint:endPoint]
reAdj = reAdj[beginPoint:endPoint,beginPoint:endPoint]

origin_g = adj2graph(origin_adj,beginPoint)
print('\t',len(origin_g.nodes()),len(origin_g.edges()))
re_g = adj2graph(reAdj,beginPoint)
print('\t',len(re_g.nodes()),len(re_g.edges()))

graph_feature_file = open(path+"%s_graph_feature.csv"%graph_name,'a')
graph_feature_file.write("%d,%d,%d,%d\n"%(len(origin_g.nodes()), len(origin_g.edges()), len(re_g.nodes()), len(re_g.edges())))
# graph_feature_file.write(len(origin_g.nodes())+'\t'+len(origin_g.edges())+'\t'+len(re_g.nodes())+'\t'+len(re_g.edges())+'\n')
graph_feature_file.close()

if not os.path.exists(path+"pos.pickle"):
    pos = nx.spring_layout(origin_g)
    pickle.dump(pos, open(path+"pos.pickle",'wb'))
else:
    pos = pickle.load(open(path+"pos.pickle",'rb'))


# pos = nx.circular_layout(origin_g)

# draw graph
plt.figure('origin_g')
nx.draw_networkx_nodes(origin_g,pos,node_color='gray',node_size=400)
nx.draw_networkx_edges(origin_g,pos,edge_color='gray',alpha=0.8)
nx.draw_networkx_labels(origin_g,pos)
plt.savefig(path+"origin_g.png")
plt.savefig(path+"origin_g.pdf")

plt.figure('re_g')
# nx.draw(re_g,pos)
nx.draw_networkx_nodes(re_g,pos,node_color='y')
nx.draw_networkx_edges(re_g,pos,edge_color='y',alpha=0.8)
nx.draw_networkx_labels(re_g,pos)
plt.savefig(path+"re_g_%s.pdf"%cutValue)
plt.savefig(path+"re_g_%s.png"%cutValue)


# draw adj
plt.figure('origin_adj')
plt.imshow(origin_adj,cmap='Purples',interpolation='none')
plt.savefig(path+"origin_adj.png")
plt.savefig(path+"origin_adj.pdf")

plt.figure('re_adj')
plt.imshow(reAdj,cmap='Purples',interpolation='none')
plt.savefig(path+"re_adj_%s.pdf"%cutValue)
plt.savefig(path+"re_adj_%s.png"%cutValue)

# plt.show()
