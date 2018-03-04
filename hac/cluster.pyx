import networkx as nx
import heapq
import sys
from .renamemap import RenameMapping

# Use lazy errors that are PEP 8 compliant
try:
    import matplotlib.pyplot as plt
except ImportError, e:
    def plt(*args, **kwargs):
        raise e
try:
    from networkx import graphviz_layout
except ImportError:
    def graphviz_layout(*args, **kwargs):
        raise ImportError("This program needs Graphviz and either PyGraphviz or Pydot")

def nx_node_iter(graph, *args, **kwargs):
    try:
        return graph.nodes_iter(*args, **kwargs)
    except AttributeError:
        return graph.nodes(*args, **kwargs)

def nx_edge_iter(graph, *args, **kwargs):
    try:
        return graph.edges_iter(*args, **kwargs)
    except AttributeError:
        return graph.edges(*args, **kwargs)

def nx_values_iter(node):
    try:
        # Python 2 & 3
        try:
            return node.iter_values()
        except AttributeError:
            return node.values()
    except AttributeError:
        return node

cdef tuple py_int_types():
    if sys.version_info[0] == 2:
        return (int, long,)
    else:
        return (int,)

cpdef set setify(elems):
    if isinstance(elems, set):
        return elems
    return set(elems)

cpdef float weight(edge):
    if isinstance(edge, (int, float)):
        return edge
    elif isinstance(edge, dict):
        return edge.get('weight', 1.0)
    else:
        return 1.0

cpdef float weighted_edge_count(graph):
    # 2 * graph.number_of_edges()
    cdef float edges = 0.0
    for _e1, _e2, d in nx_edge_iter(graph, data=True):
        edges += weight(d)
    if edges <= 0.00000001:
        return 1.0
    return 2.0 * edges

cpdef set remove_orphans(graph, ignored=None):
    orphans = set()
    for node in graph:
        if not graph.degree(node) and (not ignored or node not in ignored):
            orphans.add(node)
    graph.remove_nodes_from(orphans)
    return orphans

cpdef int max_int_elem(graph):
    # We just care about non-zero max values
    cdef int max_int = 0
    for node in nx_node_iter(graph):
        if isinstance(node, int):
            if node > max_int:
                max_int = node
    return max_int

def int_graph_mapping(graph):
    cdef dict mapping_to_int = {}
    cdef dict mapping_to_orig = {}
    cdef int node_index = 0
    for node_index, node in enumerate(nx_node_iter(graph)):
        mapping_to_int[node] = node_index
        mapping_to_orig[node_index] = node
    return RenameMapping(mapping_to_int, mapping_to_orig, node_index)

cdef class GreedyAgglomerativeClusterer(object):
    '''
    Greedy Agglomerative Clustering
    This is a greedy agglomerative hierarchical clustering algorithm.
    The algorithm efficiently clusters large number of nodes and is one of the best scaling
    clustering algorithms available. It relies on building and slicing a dendrogram of potential
    clusters from the base of a networkx graph. Each possible pairing of elements is evaluated
    and clustering in quality (see paper reference) increasing order. The greedy aspect of this
    approach is in the avoidance of backtracking. Each pass on the dengrogram assume prior passes
    were the global minimum for overall quality. Given decent edge associations, this is a
    relatively safe assumption to make and vastly increases the speed of the algorithm.

    See papers on scaling and accuracy questions regarding greedy Newman.
    '''
    cdef public int optimal_clusters
    cdef list forced_clusters
    cdef set forced_nodes
    cdef set orphans
    cdef object rename_map
    cdef object super_graph
    cdef object dendrogram_graph
    cdef list pair_cost_heap
    cdef list quality_history
    cdef int den_num

    def __init__(self, num_clusters=None):
        '''
        TODO add more configuration options to the clusterer

        Parameters:
        num_clusters : int, default=None
            The number of clusters to compute. Default to auto-detection of optimal number.
        '''
        self.optimal_clusters = num_clusters or 0

    def __getstate__(self):
        return {
            'optimal_clusters': self.optimal_clusters
        }
    getstate = __getstate__

    def __setstate__(self, state):
        self.optimal_clusters = state['optimal_clusters']
    setstate = __setstate__

    def __reduce__(self):
        # Python 2 insists on using __reduce__ for cython objects...
        return (self.__class__, (self.optimal_clusters, ))

    def __hash__(self):
        return hash(self.optimal_clusters)

    def __richcmp__(self, other, cmp_op):
        if self is other:
            return True
        elif not isinstance(other, GreedyAgglomerativeClusterer):
            return False
        elif cmp_op == 2:
            return self.__eq(other)
        elif cmp_op == 3:
            return not self.__eq(other)
        return False

    cdef bint __eq(self, GreedyAgglomerativeClusterer other):
        return self.get_state() == other.get_state()

    def cluster(self, graph, forced_clusters=None):
        '''
        Performs clustering analysis to generate a dendrogram object on the graph

        Parameters:
        graph : networkx.graph
            The networkx graph to be clustered.

        forced_cluster : iterable<set>, default=None
            A fixed set of graph associations to not cluster.
            Useful for adding pre-computed clusters.
        '''
        self.forced_clusters = list(map(setify, forced_clusters or []))
        self.forced_nodes = set(node for cluster in self.forced_clusters for node in cluster)
        # TODO use sparse matrix representation?
        self.super_graph = graph.copy()
        # TODO change to separating into connected components
        self.orphans = remove_orphans(self.super_graph, self.forced_nodes)
        # TODO do better than remapping
        self.rename_map = int_graph_mapping(self.super_graph)
        nx.relabel_nodes(self.super_graph, self.rename_map.integer, copy=False)

        self.dendrogram_graph = nx.Graph()
        self.pair_cost_heap = []
        self.quality_history = []
        self.den_num = max(max_int_elem(graph), graph.number_of_nodes()) + 1

        num_edges = weighted_edge_count(self.super_graph)
        quality = 0.0
        for cluster_id, data in self.super_graph.nodes(data=True):
            # node_degree = self.super_graph.degree(cluster_id)
            node_degree = 0.0
            for edge in nx_values_iter(self.super_graph[cluster_id]):
                node_degree += weight(edge)

            average_degree = float(node_degree) / (num_edges or 1.0)
            self.super_graph.add_node(cluster_id, degree=average_degree)
            self.dendrogram_graph.add_node(cluster_id, **data)
            # From equation (1) in section II of the Newman paper
            quality -= float(node_degree * node_degree) / (num_edges * num_edges)

        for (cluster_id1, cluster_id2, edge) in self.super_graph.edges(data=True):
            edge_weight = weight(edge) / num_edges
            self.super_graph[cluster_id1][cluster_id2]['weight'] = edge_weight
            self.super_graph[cluster_id2][cluster_id1]['weight'] = edge_weight

        self.reheapify()
        if self.forced_clusters:
            self.build_forced_clusters()
        self.run_greedy_clustering(quality)

        return Dendrogram(self.dendrogram_graph, self.quality_history, self.orphans,
            self.rename_map, self.optimal_clusters)

    def build_forced_clusters(self):
        # create a cluster from "unspecified" nodes
        for cluster in self.forced_clusters:
            precluster_nodes = []
            for node in self.rename_map.integer.iterkeys():
                if node in cluster:
                    precluster_nodes.append(self.rename_map.integer[node])
            while len(precluster_nodes) > 1:
                self.combine_clusters(precluster_nodes[0], precluster_nodes[1])
                precluster_nodes.pop(0)
                precluster_nodes.pop(0)
                while precluster_nodes:
                    self.combine_clusters(self.den_num-1, precluster_nodes[0])
                    precluster_nodes.pop(0)

    def reheapify(self):
        self.pair_cost_heap = []
        for (id1, id2) in nx_edge_iter(self.super_graph):
            self.add_pair_to_cost_heap(id1, id2)

    def run_greedy_clustering(self, quality, reheap_steps=500):
        self.quality_history = [quality]
        last_heapify = self.super_graph.number_of_nodes()
        while len(self.super_graph) > 1:
            while True:
                if self.pair_cost_heap:
                    qd, id1, id2 = heapq.heappop(self.pair_cost_heap)
                else:
                    for x in self.super_graph.nodes():
                        # combining nodes can cause x to be removed before
                        # iteration completes, so need to check its existence
                        if self.super_graph.has_node(x):
                            if not self.super_graph[x]:
                                self.combine_clusters(x, max(self.super_graph.nodes()))
                                self.quality_history.append(quality)
                    break
                if self.super_graph.has_node(id1) and self.super_graph.has_node(id2):
                    qual_diff = -qd
                    break
            if self.super_graph.number_of_edges() > 0:
                quality += qual_diff
                self.combine_clusters(id1, id2)
                self.quality_history.append(quality)

    def add_pair_to_cost_heap(self, id1, id2):
        qd = self.quality_difference(id1, id2)
        if id2 < id1:
            id1, id2 = id2, id1
        # Negate quality difference (to maximize), AND id1 < id2
        heapq.heappush(self.pair_cost_heap, (-qd, id1, id2))

    def quality_difference(self, cluster_id1, cluster_id2):
        # The "Change in Q" as described by section II of the Newman paper
        cdef float degree_one = self.super_graph.node[cluster_id1]['degree']
        cdef float degree_two = self.super_graph.node[cluster_id2]['degree']
        cdef float edge = weight(self.super_graph[cluster_id1][cluster_id2])
        return 2.0 * (edge - degree_one * degree_two)

    def combine_clusters(self, cluster_id1, cluster_id2):
        cdef int combine_id = self.den_num
        cdef float total
        self.den_num += 1

        # Add combined node
        c1_con = self.super_graph[cluster_id1]
        c2_con = self.super_graph[cluster_id2]
        cdef set c12_nodes = set(c1_con.keys()).union(set(c2_con.keys()))

        self.rename_map.original[combine_id] = combine_id
        self.rename_map.integer[combine_id] = combine_id

        cdef float degree_one = self.super_graph.node[cluster_id1]['degree']
        cdef float degree_two = self.super_graph.node[cluster_id2]['degree']
        self.super_graph.add_node(combine_id, degree=degree_one + degree_two)

        for outer_node in c12_nodes:
            total = 0.0
            # ignore edges between the two clusters
            if outer_node == cluster_id1 or outer_node == cluster_id2:
                continue
            # sum edge weights to clusters that are reached by both clusters
            if outer_node in c1_con:
                total += weight(c1_con[outer_node])
            if outer_node in c2_con:
                total += weight(c2_con[outer_node])
            self.super_graph.add_edge(combine_id, outer_node, weight=total)
            self.super_graph.add_edge(outer_node, combine_id, weight=total)
            self.add_pair_to_cost_heap(combine_id, outer_node)

        # Remove old nodes
        # TODO the except should be removed and the initial weights bug solved...
        try: self.super_graph.remove_node(cluster_id1)
        except nx.exception.NetworkXError: pass
        try: self.super_graph.remove_node(cluster_id2)
        except nx.exception.NetworkXError: pass

        # Update dendrogram
        self.dendrogram_graph.add_node(combine_id)
        self.dendrogram_graph.add_edge(combine_id, cluster_id1)
        self.dendrogram_graph.add_edge(combine_id, cluster_id2)

cdef class Dendrogram(object):
    cdef public int optimal_clusters
    cdef public list quality_history
    cdef public int max_clusters
    cdef public object int_graph
    cdef set orphans
    cdef object rename_map

    def __init__(self, dendrogram_graph, quality_history, orphans, rename_map,
            num_clusters=None):
        self.int_graph = dendrogram_graph
        self.quality_history = quality_history
        self.orphans = orphans
        self.rename_map = rename_map
        self.max_clusters = rename_map.max_node + 1 + len(orphans)
        self.optimal_clusters = num_clusters

    def __getstate__(self):
        return {
            'optimal_clusters': self.optimal_clusters,
            'quality_history': self.quality_history,
            'max_clusters': self.max_clusters,
            'int_graph': self.int_graph,
            'orphans': self.orphans,
            'rename_map': self.rename_map
        }
    getstate = __getstate__

    def __setstate__(self, state):
        self.optimal_clusters = state['optimal_clusters']
        self.quality_history = state['quality_history']
        self.max_clusters = state['max_clusters']
        self.int_graph = state['int_graph']
        self.orphans = state['orphans']
        self.rename_map = state['rename_map']
    setstate = __setstate__

    def __reduce__(self):
        # Python 2 insists on using __reduce__ for cython objects...
        return (self.__class__, (self.int_graph, self.quality_history,
            self.orphans, self.rename_map, self.optimal_clusters))

    def __hash__(self):
        return hash(self.int_graph)

    def __richcmp__(self, other, cmp_op):
        if self is other:
            return True
        elif not isinstance(other, Dendrogram):
            return False
        elif cmp_op == 2:
            return self.__eq(other)
        elif cmp_op == 3:
            return not self.__eq(other)
        return False

    cdef bint __eq(self, Dendrogram other):
        return self.get_state() == other.get_state()

    def crawl(self, start, priors=None, max_fringe_size=None):
        if priors is None:
            priors = set()
        fringe = []
        priors.add(start)
        heapq.heappush(fringe, -start)

        while len(fringe) > 0 and (max_fringe_size == None or len(fringe) < max_fringe_size):
            node = -heapq.heappop(fringe)
            priors.add(node)
            for inode in self.int_graph[node]:
                if inode not in priors:
                    heapq.heappush(fringe, -inode)

        return priors, fringe

    cdef int choose_num_clusters(self, num_clusters):
        cdef bint valid_choice = isinstance(num_clusters, py_int_types())
        if not valid_choice:
            if self.optimal_clusters > 0:
                num_clusters = self.optimal_clusters
            else:
                index, value = max(enumerate(self.quality_history), key=lambda iv: iv[1])
                num_clusters = len(self.quality_history) - index
        return max(min(num_clusters, self.max_clusters), 0)

    def clusters(self, num_clusters=None):
        cdef int picked_num_cluster = self.choose_num_clusters(num_clusters)
        cdef list clusters = [set([n]) for n in self.orphans]
        if self.int_graph and picked_num_cluster > 0:
            start_node = max(self.int_graph)
            priors, fringe = self.crawl(start=start_node, max_fringe_size=picked_num_cluster)

            # Double check we got the right number of values
            if len(fringe) != picked_num_cluster:
                raise ValueError("Failed to retrieve %d clusters correctly (got %d instead)"
                    % (picked_num_cluster, len(fringe)))

            for neg_clust_start in fringe:
                clust_start = -neg_clust_start
                cprior, cfringe = self.crawl(start=clust_start, priors=priors.copy())
                cluster_set = set()
                for node in cprior:
                    if (node <= clust_start and
                        node <= self.rename_map.max_node and
                        node in self.rename_map.original):
                        cluster_set.add(self.rename_map.original[node])
                if cluster_set:
                    clusters.append(cluster_set)
        return sorted(clusters, key=lambda c: -len(c))

    def labels(self, num_clusters=None):
        cdef dict labels = {}
        for label, nodes in enumerate(self.clusters(num_clusters)):
            for node in nodes:
                labels[node] = label
        return labels

    def plot_quality_history(self, plot_name, out_file_name, show=True):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.quality_history), 0, -1), self.quality_history)
        plt.title('Number of Clusters vs. Modularity for %s' % plot_name)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Modularity Score')
        plt.savefig(out_file_name)
        if show:
            plt.show()

    def plot(self, filename, figure_size=(10,10), font_size=10, show=True):
        graph = nx.relabel_nodes(self.int_graph, self.rename_map.original, copy=False)
        pos = graphviz_layout(graph, prog='twopi', args='')
        plt.figure(figsize=figure_size)
        nx.draw(graph, pos, node_size=10, font_size=font_size, alpha=0.5,
                node_color="blue", with_labels=True)
        plt.axis('equal')
        plt.savefig(filename)
        if show:
            plt.show()
