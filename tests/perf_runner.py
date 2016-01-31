import networkx as nx
import random
import time
import sys
from hac import GreedyAgglomerativeClusterer

if __name__ == '__main__':
    size = 1000
    edge_size = 6 # 2*log(size)
    graph = nx.Graph()
    print "Adding nodes..."
    graph.add_nodes_from(xrange(size))
    print "Adding edges..."
    edges = []
    for node in xrange(size):
        for rand_node in random.sample(xrange(size), edge_size):
            edges.append((node, rand_node))
    graph.add_edges_from(edges)

    print "Starting Clustering on ({} nodes, {} edges) ....".format(
        graph.number_of_nodes(), graph.number_of_edges())
    sys.stdout.flush()
    start = time.clock()
    GreedyAgglomerativeClusterer().cluster(graph).clusters()
    print "Finished Clustering in {} seconds".format((time.clock() - start))
