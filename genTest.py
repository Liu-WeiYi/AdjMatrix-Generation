####################################
# Sailung Yeung <yeungsl@bu.edu>
# parse the graph into desired form
#
####################################
import networkx as nx
import os, sys

def gen(path):
    n = 500
    k = 2
    p = 0.2
    WS = nx.watts_strogatz_graph(n, k ,p)
    write(WS, path, 'WS')
    BA = nx.barabasi_albert_graph(n,k)
    write(BA, path, 'BA')
    ER = nx.erdos_renyi_graph(n,p)
    write(ER, path, 'ER')

def write(graph, path, name):
    name_path = os.path.join(path, name, '')
    if not os.path.exists(name_path):
        print("CREATING NEW DIR-----------------", name_path)
        os.mkdir(name_path)
    w_file = open(name_path+'%s.txt'%name, 'w+')
    for e in graph.edges():
        w_file.write(str(e[0]) + '\t' + str(e[1]) + '\n')
    w_file.close()

if __name__ == "__main__":
    path = sys.argv[1]
    gen(path)
    print('---------------------------done!')
