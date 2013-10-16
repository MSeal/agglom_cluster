agglom_cluster
==============
Agglomerative clustering tool for network-x graphs

## Clustering
Implements the algorithm described by:
"Fast algorithm for detecting community structure in networks"
M. E. J. Newman. 2004
http://arxiv.org/pdf/cond-mat/0309508v1.pdf
This is a greedy agglomerative hierarchical clustering algorithm

This implementation uses a heap to select the best pair to cluster at each iteration
- A naive implementation considers all "n" edges in the graph (O(n))
- A heap reduces this search dramatically (O(log(n))

## Problems
1) The actual Modularity score does not exactly match the Modularity score of the example on the wikipedia page
   - http://en.wikipedia.org/wiki/Modularity_(networks)
2) Does not work for directed graphs
3) Does not work for negative graphs (TODO add this capability)

Stores the following information
1) Supergraph
   - Duplicate of the original graph. Gets manipulated during the clustering: edges and nodes are condensed and reweighted
   - Cluster degree stored in node attribute
   - Number of connections between clusters stored in edge attribute (weighted edge)
   - Implicitly keeps track of the existing nodes which is required for the heap
2) Dendrogram
   - Stores the clustering history in a tree
3) Heap
   - Stores the modularity quality difference for combining each pair of existing nodes
   - Popping returns the node pair with the largest modulairty/quality difference, then smallest id1, then smallest id2
       * Stored as a tuple (value, id1, id2) where the modularity/value is negated, and id1 is the smaller of the two
       * Processing the smaller IDs first means that smaller clusters will be chosen first during modularity tie breaking
   - Non-existing nodes are not actively removed from the heap, so pairs with non-existing nodes are ignored when popping
4) Quality History
   - Charts the Modularity score for each number of clustering

## Author
Author(s): Ethan Lozano & Matthew Seal

Collaborator(s):

#### (C) Copyright 2013, [Opengov](http://opengov.com)
