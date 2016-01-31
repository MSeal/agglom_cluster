[![Build Status](https://travis-ci.org/MSeal/agglom_cluster.svg?branch=master)](https://travis-ci.org/MSeal/agglom_cluster)

# hac
Agglomerative clustering tool for network-x graphs

## Clustering
Implements the algorithm described by:
"Fast algorithm for detecting community structure in networks"
M. E. J. Newman. 2004
http://arxiv.org/pdf/cond-mat/0309508v1.pdf
The algorithm efficiently clusters large number of nodes and is one of the best scaling clustering algorithms available. It relies on building and slicing a dendrogram of potential clusters from the base of a networkx graph. Each possible pairing of elements is evaluated and clustering in quality (see paper reference) increasing order. The greedy aspect of this approach is in the avoidance of backtracking. Each pass on the dengrogram assume prior passes were the global minimum for overall quality. Given decent edge associations, this is a relatively safe assumption to make and vastly increases the speed of the algorithm.

See papers on scaling and accuracy questions regarding greedy Newman.

This implementation uses a heap to select the best pair to cluster at each iteration
- A naive implementation considers all "n" edges in the graph (O(n))
- A heap reduces this search dramatically (O(log(n))

## Installation

    pip install agglomcluster

## Dependencies
networkx -- supported graphing library

## Examples

    clusterer = GreedyAgglomerativeClusterer()
    # This cluster call is where most of the heavy lifting happens
    karate_dendrogram = clusterer.cluster(nx.karate_club_graph())
    karate_dendrogram.clusters(1)
    # => [set(range(34))]

    karate_dendrogram.clusters(2)
    # => [set([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 21]),
          set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])]

    karate_dendrogram.clusters(3)
    # => [set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
          set([1, 2, 3, 7, 9, 12, 13, 17, 21]),
          set([0, 4, 5, 6, 10, 11, 16, 19])]

    # We can ask the dendrogram to pick the optimal number of clusters
    karate_dendrogram.clusters()
    # => [set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
          set([1, 2, 3, 7, 9, 12, 13, 17, 21]),
          set([0, 4, 5, 6, 10, 11, 16, 19])]

    karate_dendrogram.labels()
    # => { 0: 2, 1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2, 7: 1, 8: 0, 9: 1, 10: 2, 11: 2,
          12: 1, 13: 1, 14: 0, 15: 0, 16: 2, 17: 1, 18: 0, 19: 2, 20: 0, 21: 1, 22: 0,
          23: 0, 24: 0, 25: 0, 26: 0, 27: 0, 28: 0, 29: 0, 30: 0, 31: 0, 32: 0, 33: 0 }

    # We can also force certain nodes to always be clustered together
    forced_clusters = [set([33,0]), set([32,1])]
    forced_karate_dendrogram = clusterer.cluster(nx.karate_club_graph(), forced_clusters=forced_clusters)
    forced_karate_dendrogram.clusters()
    # => [set([0, 33, 9, 11, 12, 14, 15, 17, 18, 19, 21, 26, 29]),
          set([32, 1, 2, 3, 7, 8, 13, 20, 22, 30]),
          set([23, 24, 25, 27, 28, 31]),
          set([16, 10, 4, 5, 6])]

## Issues
* The actual Modularity score does not exactly match the Modularity score of the example on the wikipedia page (extensive use and investigation indicates it's not affecting the quality of results, just makes it difficult to match referenced paper's exact results)
   - http://en.wikipedia.org/wiki/Modularity_(networks)
* Does not handle disconnected components (unless they are components of size 1)
* Node relabeling is messy (adds hashable nodes dependency)
* Dendrogram crawling is used for two separate purposes which aren't clearly defined/called

## Limitations:
* Nodes inside clustered graph must be hashable elements
* Does not work for directed graphs (TODO operate on the undirected graph)
* Does not work for negative graphs (TODO add this capability)

## TODO
* Move issues to github issues and out of README
* Consider using a scikit sparse matrix for the dengrogram generation as an optimization
* Convert clustering process into a multi-thread/process capable version
* Consider interface/capabilty parity with scikit AgglomerativeCluster
* Add evaluation function options to clusterer other than originally defined quality
* A few methods could use documentation

## Classes
### GreedyAgglomerativeClusterer
Used to generate Dendrogram objects that represent a clustered graph. Use `.cluster()` to process a graph.

### Dendrogram
The clustered result from an agglomerative clustering pass. Use `.clusters()` and `.labels()` to get the desired cluster results. Additionally you this class has some built-in graphing methods `.plot()` and `.plot_quality_history()`.

## Performance

Approximate performance runs on natural graph sizes on high-end machine:

    Nodes     | Edges     | Time      | Memory
    1000      | 6000      | 1.5 s     | 28 MB
    10000     | 80000     | 350 s     | 2.5 GB
    TODO More sizes

## Author
Author(s): Matthew Seal
Past Author/Contributors(s): Ethan Lozano, Zubin Jelveh
