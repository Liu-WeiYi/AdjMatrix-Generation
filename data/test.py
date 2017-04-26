# coding: utf-8

import networkx as nx

g = nx.watts_strogatz_graph(300, 2, 0.2)

g_file = open('WS2.edge_list','w+')

for edge in g.edges():
    src,dst = edge
    g_file.write(str(src)+"\t"+str(dst)+"\n")

g_file.close()