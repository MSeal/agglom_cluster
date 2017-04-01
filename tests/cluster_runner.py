# This import fixes sys.path issues
from . import parentpath

import os
import networkx as nx
from hac import GreedyAgglomerativeClusterer

def main():
    graph = nx.karate_club_graph();
    dendrogram = GreedyAgglomerativeClusterer().cluster(graph)
    print(dendrogram.quality_history)
    print(dendrogram.clusters())
    try:
        dendrogram.plot(os.path.join(os.path.dirname(__file__), '..', 'pics', 'karate_dend.png'), show=False)
        dendrogram.plot_quality_history('Karate', os.path.join(os.path.dirname(__file__), '..', 'pics', 'karate'), show=False)
    except:
        pass

if __name__ == '__main__':
    main()
