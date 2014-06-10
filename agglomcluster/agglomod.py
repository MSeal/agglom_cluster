import networkx as nx
import heapq
import os
import pickle
from collections import namedtuple

def _get_plot_libs():
    import matplotlib.pyplot as plt
    try:
        from networkx import graphviz_layout
    except ImportError:
        raise ImportError("This program needs Graphviz and either PyGraphviz or Pydot")
    return plt, graphviz_layout

class NewmanGreedy:

    RenameMapping = namedtuple('RenameMapping', ['integer', 'original'])

    def __init__(self, graph, snapshot_size=None, forced_clusters=None, copy_original=True):
        if copy_original:
            graph = graph.copy()

        self.forced_clusters = [set(cluster) for cluster in forced_clusters] if forced_clusters else []
        # TODO change to separating into connected components
        self.orphans = self.remove_orphans(graph)

        self.rename_map = self.remap(graph)
        nx.relabel_nodes(graph, self.rename_map.integer, copy=False)

        self.orig = graph
        self.max_clusters = len(graph)
        self.super_graph = graph.copy()
        self.dendrogram = nx.Graph()
        self.snapshot_size = snapshot_size
        self.pair_cost_heap = []
        self.quality_history = []
        self.den_num = self.super_graph.number_of_nodes()
        for node in graph.node:
            if (isinstance(node, int)) and node > self.den_num:
                self.den_num = node + 1

        def e_weight(graph, e):
            edge = graph[e[0]][e[1]]
            if isinstance(edge, int):
                return edge
            elif isinstance(edge, dict) and 'weight' in edge:
                return edge['weight']
            else:
                return 1
        num_edges = 2 * sum(e_weight(graph, e) for e in graph.edges_iter()) #2*graph.number_of_edges()

        quality = 0.0
        for cluster_id in graph.nodes_iter():
            #node_degree = graph.degree(cluster_id)
            node_degree = 0
            for tags in graph[cluster_id].itervalues():
                if isinstance(tags, dict) and 'weight' in tags:
                    node_degree += tags['weight']
                else:
                    node_degree += 1

            ai = float(node_degree) / num_edges
            self.super_graph.node[cluster_id] = ai
            self.dendrogram.add_node(cluster_id)
            # From equation (1) in secion II of the Newman paper
            quality -= float(node_degree * node_degree) / (num_edges * num_edges)

        for (cluster_id1, cluster_id2) in graph.edges_iter():
            if isinstance(graph[cluster_id1][cluster_id2], (int, float)):
                old_e_ij = graph[cluster_id1][cluster_id2]
            elif isinstance(graph[cluster_id1][cluster_id2], dict) and 'weight' in graph[cluster_id1][cluster_id2]:
                old_e_ij = graph[cluster_id1][cluster_id2]['weight']
            else:
                old_e_ij = 1.0
            eij = float(old_e_ij) / num_edges
            self.super_graph[cluster_id1][cluster_id2] = eij
            self.super_graph[cluster_id2][cluster_id1] = eij
        # Could be calcuated in the edge for-loop above at a loss of code reuse or efficiency
        #for (id1, id2) in graph.edges_iter():
        #    self.add_pair_to_cost_heap(id1, id2)
        self.reheapify()
        if self.forced_clusters:
            self.build_forced_clusters()
        self.run_greedy_clustering(quality)
        nx.relabel_nodes(self.dendrogram, self.rename_map.original, copy=False)

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

    def remove_orphans(self, graph):
        # remove orphan nodes except those in "unspecified" cluster
        orphans = [node for node in graph if 
                (not graph.degree(node) and not any(node in cluster for cluster in self.forced_clusters))]
        graph.remove_nodes_from(orphans)
        return orphans

    def remap(self, graph):
        mapping_to_int = {}
        mapping_to_orig = {}
        for node_index, node in enumerate(graph.nodes_iter()):
            mapping_to_int[node]= node_index
            mapping_to_orig[node_index] = node
        return self.RenameMapping(mapping_to_int, mapping_to_orig)

    def reheapify(self):
        del self.pair_cost_heap
        self.pair_cost_heap = []
        for (id1, id2) in self.super_graph.edges_iter():
            self.add_pair_to_cost_heap(id1, id2)

    def run_greedy_clustering(self, quality, reheap_steps=500):
        self.quality_history = [quality]
        if self.snapshot_size == None or self.snapshot_size > self.super_graph.number_of_nodes():
            self.snapshot = self.super_graph.copy()
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
                if(self.super_graph.has_node(id1) and self.super_graph.has_node(id2)):
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

    # The "Change in Q" as described by section II of the Newman paper
    def quality_difference(self, cluster_id1, cluster_id2):
        ai = float(self.super_graph.node[cluster_id1])
        aj = float(self.super_graph.node[cluster_id2])
        eij = float(self.super_graph[cluster_id1][cluster_id2])
        return 2.0*(eij - ai*aj)

    def combine_clusters(self, cluster_id1, cluster_id2):
        combine_id = self.den_num
        self.den_num += 1
        
        # Add combined node
        c1_con = self.super_graph[cluster_id1]
        c2_con = self.super_graph[cluster_id2]
        c12_nodes = set(c1_con.keys()).union(set(c2_con.keys()))

        self.rename_map.original[combine_id] = combine_id
        self.rename_map.integer[combine_id] = combine_id
        
        self.super_graph.add_node(combine_id)
        combined_degree = self.super_graph.node[cluster_id1] + self.super_graph.node[cluster_id2]
        self.super_graph.node[combine_id] = combined_degree
        for outer_node in c12_nodes:
            total = 0.0
            # ignore edges between the two clusters
            if(outer_node == cluster_id1 or outer_node == cluster_id2):
                continue
            # sum edge weights to clusters that are reached by both clusters
            if(outer_node in c1_con):
                total += c1_con[outer_node]
            if(outer_node in c2_con):
                total += c2_con[outer_node]
            #self.super_graph.add_edge(combine_id, outer_node)
            #self.super_graph.add_edge(outer_node, combine_id)
            self.super_graph[combine_id][outer_node] = total
            self.super_graph[outer_node][combine_id] = total
            self.add_pair_to_cost_heap(combine_id, outer_node)
        
        # Remove old nodes
        # TODO the except should be removed and the initial weights bug solved...
        try: self.super_graph.remove_node(cluster_id1)
        except nx.exception.NetworkXError: pass
        try: self.super_graph.remove_node(cluster_id2)
        except nx.exception.NetworkXError: pass
        
        # Update dendrogram
        self.dendrogram.add_node(combine_id)
        self.dendrogram.add_edge(combine_id, cluster_id1)
        self.dendrogram.add_edge(combine_id, cluster_id2)

    def dendrogram_crawl(self, start, priors=None, max_fringe_size=None):
        if priors == None:
            priors = set()
        fringe = []

        # Helper function to push node neighbors into fringe
        def push_dend_list(fringe, priors, node):
            priors.add(node)
            for inode in self.dendrogram[node]:
                if inode not in priors:
                    heapq.heappush(fringe, -inode)

        priors.add(start)
        heapq.heappush(fringe, -start)

        step = 0
        while len(fringe) > 0 and (max_fringe_size == None or len(fringe) < max_fringe_size):
            node = -heapq.heappop(fringe)
            push_dend_list(fringe, priors, node)
            step += 1

        return priors, fringe

    def get_clusters(self, num_clusters=None):
        if num_clusters == None:
            index, value = max(enumerate(self.quality_history), key=lambda iv: iv[1])
            num_clusters = len(self.quality_history) - index
        num_clusters = max(min(num_clusters, self.max_clusters), 0)

        clusters = [set([n]) for n in self.orphans]
        if self.dendrogram and num_clusters:
            nx.relabel_nodes(self.dendrogram, self.rename_map.integer, copy=False)
            try:
                start_node = max(self.dendrogram)
                priors, fringe = self.dendrogram_crawl(start=start_node, max_fringe_size=num_clusters)

                # Double check we got the right number of values
                if len(fringe) != num_clusters:
                    raise ValueError("Failed to retrieve %d clusters correctly (got %d instead)" 
                        % (num_clusters, len(fringe)))

                for neg_clust_start in fringe:
                    clust_start = -neg_clust_start
                    cprior, cfringe = self.dendrogram_crawl(start=clust_start, priors=priors.copy())
                    cluster_set = set(self.rename_map.original[n] for n in cprior
                                      if n <= clust_start and self.orig.has_node(n))
                    if cluster_set:
                        clusters.append(cluster_set)
            finally:
                nx.relabel_nodes(self.dendrogram, self.rename_map.original, copy=False)
        return sorted(clusters, key=lambda c: -len(c))

    def get_super_graph(self, size=None):
        if size == None:
            return self.super_graph
        
        clusters = self.get_clusters(size)
        #TODO recombine nodes...

    def plot_quality_history(self, plot_name, out_file_name, show=True):
        plt, graphviz_layout = _get_plot_libs()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(range(len(self.quality_history), 0, -1), self.quality_history)
        plt.title('Number of Clusters vs. Modularity for %s' % plot_name)
        ax.set_xlabel('Number of Clusters')
        ax.set_ylabel('Modularity Score')
        plt.savefig(out_file_name)
        if show:
            plt.show()

    def plot_dendrogram(self, filename, figure_size=(10,10), font_size=10, show=True):
        plt, graphviz_layout = _get_plot_libs()
        pos = graphviz_layout(self.dendrogram, prog='twopi', args='')
        plt.figure(figsize=figure_size)
        nx.draw(self.dendrogram, pos, node_size=10, font_size=font_size, alpha=0.5,
                node_color="blue", with_labels=True)
        plt.axis('equal')
        plt.savefig(filename)
        if show:
            plt.show()

    @staticmethod
    def build_load(graph, graph_name, regen_clustering=False, snapshot_size=None):
        name, ext = os.path.splitext(graph_name)
        pkl_file_name = name + '.pkl'
        try:
            if regen_clustering:
                raise Exception("Skip loading")
            with open(pkl_file_name, 'rb') as pkl_file:
                cluster_graph = pickle.load(pkl_file)
        except:
            cluster_graph = NewmanGreedy(graph, snapshot_size=snapshot_size)
            with open(pkl_file_name, 'wb') as pkl_file:
                pickle.dump(cluster_graph, pkl_file)
        return cluster_graph

def main():
    graph = nx.karate_club_graph();
    newman = NewmanGreedy(graph)
    print newman.quality_history
    print newman.get_clusters()
    try:
        newman.plot_dendrogram(os.path.join(os.path.dirname(__file__), '..', 'pics', 'karate_dend.png'), show=False)
        newman.plot_quality_history('Karate', os.path.join(os.path.dirname(__file__), '..', 'pics', 'karate'), show=False)
    except:
        pass
    
if __name__ == '__main__':
    main()
