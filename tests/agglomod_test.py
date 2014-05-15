# This import fixes sys.path issues
import parentpath

import unittest
import networkx as nx
from agglomcluster import NewmanGreedy

class GreedyAgglomTest(unittest.TestCase):
    def karate_clustering(self, *args, **kwargs):
        return NewmanGreedy(nx.karate_club_graph(), *args, **kwargs)

    def test_quality_history(self):
        self.assertListEqual(self.karate_clustering().quality_history,
            [-0.04980276134122286, -0.03763971071663378, -0.013971071663379343, -0.001479289940828389, 0.017751479289940843, 
             0.04906311637080869, 0.06122616699539778, 0.0784845496383958, 0.09467455621301776, 0.11111111111111112, 
             0.12319197896120973, 0.14587442472057857, 0.1577087442472058, 0.16904996712689022, 0.1802268244575937, 
             0.19140368178829717, 0.2081689677843524, 0.21918145956607496, 0.22937212360289283, 0.24317882971729124, 
             0.25895792241946086, 0.27925706771860614, 0.28944773175542404, 0.29930966469428005, 0.3111439842209073, 
             0.3205128205128205, 0.32955292570677186, 0.33826429980276135, 0.34935897435897434, 0.3628369493754109, 
             0.37598619329388555, 0.38067061143984215, 0.37179487179487175, -5.551115123125783e-17])

    def test_clustering(self):
        karate = self.karate_clustering()
        self.assertListEqual(karate.get_clusters(1), [set(range(34))])

        self.assertListEqual(karate.get_clusters(2),
            [set([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 21]),
             set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])])

        self.assertListEqual(karate.get_clusters(3),
            [set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]), 
             set([1, 2, 3, 7, 9, 12, 13, 17, 21]),
             set([0, 4, 5, 6, 10, 11, 16, 19])])

        # Requesting more clusters than are available returns the maximum possible instead
        for num_clusters in range(34, 40):
            self.assertListEqual(sorted(karate.get_clusters(num_clusters), key=lambda c: iter(c).next()), 
                [set([index]) for index in range(34)])
        for num_clusters in range(0, -4, -1):
            self.assertListEqual(karate.get_clusters(num_clusters), [])

    def test_orphans(self):
        graph = nx.Graph();
        graph.add_nodes_from(range(10))
        graph.add_edges_from([[0, 1], [1, 2], [0, 2]])
        newman = NewmanGreedy(graph)

        clusters = newman.get_clusters()
        self.assertIn(set([0, 1, 2]), clusters)
        for node in range(3, 10):
            self.assertIn(set([node]), clusters)

    def test_small_graph(self):
        graph = nx.Graph();
        newman = NewmanGreedy(graph)
        self.assertEqual(list(newman.get_clusters()), [])

        graph.add_node(1)
        newman = NewmanGreedy(graph)
        self.assertEqual(list(newman.get_clusters(1)[0]), [1])
        self.assertEqual(list(newman.get_clusters()[0]), [1])

    def test_forced_clusters(self):
        forced_clusters = [set([33,0]), set([32,1])]
        karate = self.karate_clustering(forced_clusters=forced_clusters)
        clusters = karate.get_clusters()

        for fc in forced_clusters:
            first_elem = iter(fc).next()
            match_cluster = None
            for cluster in clusters:
                if first_elem in cluster:
                    match_cluster = cluster
                    break
            self.assertIsNotNone(match_cluster)
            for elem in fc:
                self.assertIn(elem, match_cluster)

if __name__ == "__main__":
    unittest.main()
