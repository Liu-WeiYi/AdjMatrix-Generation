import networkx as nx

g = nx.watts_strogatz_graph(n=1000, k=2, p= 0.2)

g_file = open("WS_test_1000.edge_list",'w+')

for edge in g.edges():
    src,dst = edge
    g_file.write(str(src)+'  '+str(dst)+'\n')
