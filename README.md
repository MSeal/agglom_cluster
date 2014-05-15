# agglom_cluster
Agglomerative clustering tool for network-x graphs

## Clustering
Implements the algorithm described by:
"Fast algorithm for detecting community structure in networks"
M. E. J. Newman. 2004
http://arxiv.org/pdf/cond-mat/0309508v1.pdf
This is a greedy agglomerative hierarchical clustering algorithm. The alogorithm efficiently clusters
large number of nodes (this is one of the best scaling clustering algorithms) while producing a suggested
number of clusters. See papers on scaling and accuracy questions regarding greedy Newman.

This implementation uses a heap to select the best pair to cluster at each iteration
- A naive implementation considers all "n" edges in the graph (O(n))
- A heap reduces this search dramatically (O(log(n))

## Dependencies
allset -- for automatic module importing
networkx -- supported graphing library

## Problems
* The actual Modularity score does not exactly match the Modularity score of the example on the wikipedia page
   - http://en.wikipedia.org/wiki/Modularity_(networks)
* Does not work for directed graphs (TODO operate on the undirected graph)
* Does not work for negative graphs (TODO add this capability)
* Does not handle disconnected components (unless than are components of size 1)
* Clustering needs to move to a function call rather than an object holder (return dendrogram object)
* Node relabeling is messy
* Dendrogram crawling is used for two separate purposes which aren't clearly defined/called

## Attributes
NewmanGreedy objects store the following attributes
* Supergraph
   - Duplicate of the original graph. Gets manipulated during the clustering: edges and nodes are condensed and reweighted
   - Cluster degree stored in node attribute
   - Number of connections between clusters stored in edge attribute (weighted edge)
   - Implicitly keeps track of the existing nodes which is required for the heap
* Dendrogram
   - Stores the clustering history in a tree
* Heap
   - Stores the modularity quality difference for combining each pair of existing nodes
   - Popping returns the node pair with the largest modulairty/quality difference, then smallest id1, then smallest id2
       * Stored as a tuple (value, id1, id2) where the modularity/value is negated, and id1 is the smaller of the two
       * Processing the smaller IDs first means that smaller clusters will be chosen first during modularity tie breaking
   - Non-existing nodes are not actively removed from the heap, so pairs with non-existing nodes are ignored when popping
* Quality History
   - Charts the Modularity score for each number of clustering

## Author
Author(s): Ethan Lozano & Matthew Seal

Collaborator(s): Zubin Jelveh
