####################################
# Sailung Yeung <yeungsl@bu.edu>
# parse the graph into desired form
#
####################################
import networkx as nx
import sys, glob, random


def parse(path, filename):
    id = 0
    id_map = {}
    edge_list_file = open(glob.glob(path+"*.txt")[0], 'rb')
    g = nx.Graph(name=filename)
    for line in edge_list_file.readlines():
        src, dst = line.strip().split()
        #print(str(src), str(dst))
        g.add_nodes_from([src, dst])
        g.add_edge(src, dst)
    edge_list_file.close()
    new_f = open(path+'%s.edge_list'%filename, 'w+')
    for e in g.edges():
        src = e[0]
        dst = e[1]
        if src not in id_map.keys():
            id_map[src] = id
            id += 1
        if dst not in id_map.keys():
            id_map[dst] = id
            id += 1
        if src == dst:
            print('found repeating edges....', str(e))
            r = random.randint(0,len(g.nodes()))
            while r == id_map[dst]:
                r = random.randint(0,len(g.nodes()))
            print('replace it with .....', str(r))
            new_f.write(str(id_map[src]) + '\t' + str(r) + '\n')
        else:
            new_f.write(str(id_map[src]) + '\t' + str(id_map[dst]) + '\n')
    new_f.close()
    map_file = open(path+'%s.id_map'%filename, 'w+')
    for node in id_map:
        map_file.write(str(node) + '\t' + str(id_map[node]) + '\n')
    map_file.close()
    return

if __name__ == "__main__":
    path = sys.argv[1]
    filename = sys.argv[2]
    parse(path, filename)
    print("-----------------------done!")
