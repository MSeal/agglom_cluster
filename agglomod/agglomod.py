import networkx as nx
import heapq
import os
import pickle

def _get_plot_libs():
    import matplotlib.pyplot as plt
    try:
        from networkx import graphviz_layout
    except ImportError:
        raise ImportError("This program needs Graphviz and either PyGraphviz or Pydot")
    return plt, graphviz_layout

class NewmanGreedy:
    def __init__(self, graph, snapshot_size=None):
        graph = self.remove_orphans(graph)
        self.map = self.remap(graph)
        nx.relabel_nodes(graph,self.map[0],copy=False)
        self.orig = graph
        self.super_graph = graph.copy()
        self.dendrogram = nx.Graph()
        self.snapshot_size = snapshot_size
        self.pair_cost_heap = []
        self.quality_history = []
        self.den_num = self.super_graph.number_of_nodes()
        for node in graph.node:
            if (isinstance(node, int)) and node > self.den_num:
                self.den_num = node+1
        
        def e_weight(graph, e):
            edge = graph[e[0]][e[1]]
            if isinstance(edge, int):
                return edge
            elif isinstance(edge, dict) and 'weight' in edge:
                return edge['weight']
            else:
                return 1
        num_edges = 2*sum(e_weight(graph, e) for e in graph.edges_iter()) #2*graph.number_of_edges()
            
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
            quality -= float(node_degree*node_degree)/(num_edges*num_edges)
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
        self.run_greedy_clustering(quality)
        nx.relabel_nodes(self.dendrogram,self.map[1],copy=False)

    def remove_orphans(self,graph):
        topop=[]
        for x in graph.nodes():
            if graph[x]=={}:
                topop.append(x)
        graph.remove_nodes_from(topop)
        return graph
            
    def remap(self,graph):
        mapping_to_int={}
        mapping_to_orig={}
        for n in range(len(graph.nodes())):
            mapping_to_int[graph.nodes()[n]]=n
            mapping_to_orig[n]=graph.nodes()[n]
        return [mapping_to_int,mapping_to_orig]
        
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
        while self.super_graph.number_of_nodes() > 1:
            sumo = 0
            while sumo ==0:
                if len(self.pair_cost_heap) > 0:
                    qd, id1, id2 = heapq.heappop(self.pair_cost_heap)
                else:
                    for x in self.super_graph.nodes():
                        if x in self.super_graph.nodes():
                            if self.super_graph[x]=={}:
                                self.combine_clusters(x,max(self.super_graph.nodes()))
                                self.quality_history.append(quality)
                    sumo = 1
                if(self.super_graph.has_node(id1) and self.super_graph.has_node(id2)):
                    qual_diff = -qd
                    sumo = 1
            if self.super_graph.number_of_edges()>0:
                quality += qual_diff
                self.combine_clusters(id1, id2)
                self.quality_history.append(quality)
                
    def add_pair_to_cost_heap(self, id1, id2):
        qd = self.quality_difference(id1, id2)
        if(id2 < id1):
            temp = id1
            id1 = id2
            id2 = temp
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
        
    def _to_int_node_name(self, node, node_int_bimap):
        return node_int_bimap[node] if not isinstance(node, int) else node
        
    def _to_normal_node_name(self, node, node_int_bimap):
        return node_int_bimap[node] if isinstance(node, int) else node
        
    def dendrogram_crawl(self, start, node_int_bimap, priors=None, max_steps=None):
        if priors == None:
            priors = set()
        fringe = []
        
        # Helper function to push node neighbors into fringe
        def push_dend_list(fringe, priors, node):
            priors.add(self._to_normal_node_name(node, node_int_bimap))
            for n in self.dendrogram[self._to_normal_node_name(node, node_int_bimap)]:
                # Convert to int if we have something else
                inode = self._to_int_node_name(n, node_int_bimap)
                if inode not in priors:
                    #print "Adding", n, "From", node
                    heapq.heappush(fringe, -inode)
                    
        priors.add(start)
        heapq.heappush(fringe, -self._to_int_node_name(start, node_int_bimap))
        
        step = 0
        while len(fringe) > 0 and (max_steps == None or step < max_steps):
            node = -self._to_int_node_name(heapq.heappop(fringe), node_int_bimap)
            #print "Removing", node
            push_dend_list(fringe, priors, node)
            step += 1
            
        return priors, fringe
        
    def get_clusters(self, num_clusters=None):
        if num_clusters == None:
            index, value = max(enumerate(self.quality_history), key=lambda iv: iv[1])
            num_clusters = len(self.quality_history) - index
        
        node_int_bimap = {}
        start_node = None
        cur_int = 0
        for dn in self.dendrogram.nodes_iter():
            if isinstance(dn, int):
                if start_node == None or dn > start_node:
                    start_node = dn
                node_int_bimap[dn] = dn
            else:
                node_int_bimap[dn] = cur_int
                node_int_bimap[cur_int] = dn
                cur_int += 1
        
        priors, fringe = self.dendrogram_crawl(start=start_node,
                                               node_int_bimap=node_int_bimap,
                                               max_steps=num_clusters-1)
        # Double check we got the right number of values
        if len(fringe) != num_clusters:
            raise ValueError("get_clusters failed to retrieve "+
                "%d clusters correctly (got %d instead)" % (num_clusters, len(fringe)))
        
        clusters = []
        for neg_clust_start in fringe:
            clust_start = -neg_clust_start
            #print clust_start
            cprior, cfringe = self.dendrogram_crawl(start=clust_start,
                                                    node_int_bimap=node_int_bimap,
                                                    priors=priors.copy())
            clusters.append(set(self._to_normal_node_name(n, node_int_bimap) for n in cprior 
                                if self._to_int_node_name(n, node_int_bimap) <= clust_start))
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

    def plot_dendrogram(self,filename='karate_dendrogram.png',fsize=10):
        plt, graphviz_layout = _get_plot_libs()
        pos = graphviz_layout(self.dendrogram, prog='twopi', args='')
        plt.figure(figsize=(10,10))
        nx.draw(self.dendrogram, pos, node_size=10,font_size=fsize, alpha=0.5, 
                node_color="blue", with_labels=True)
        plt.axis('equal')
        plt.savefig(filename)
        plt.show()
        
    @staticmethod
    def build_load(graph, graph_name, regen_clustering=False, snapshot_size=None):
        name, ext = os.path.splitext(graph_name)
        pickleFName = name + '.pkl'
        try:
            if regen_clustering:
                raise Exception("Skip loading")
            with open(pickleFName, 'rb') as pklFile:
                cluster_graph = pickle.load(pklFile)
        except:
            cluster_graph = NewmanGreedy(graph, snapshot_size=snapshot_size)
            with open(pickleFName, 'wb') as pklFile:
                pickle.dump(cluster_graph, pklFile)
        return cluster_graph

def main():
    G = nx.karate_club_graph();
    N = NewmanGreedy(G)
    print N.quality_history
    try:
        N.plot_dendrogram()
        N.plot_quality_history('Karate', os.path.join(os.path.dirname(__file__), 'pics', 'karate'))
    except:
        pass
    
if __name__ == '__main__':
    main()
