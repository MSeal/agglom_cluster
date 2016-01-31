# This import fixes sys.path issues
import parentpath

import unittest
import networkx as nx
from hac.cluster import (
    GreedyAgglomerativeClusterer,
    setify,
    weight,
    weighted_edge_count,
    remove_orphans,
    int_graph_mapping)

class GreedyAgglomTest(unittest.TestCase):
    def karate_clustering(self, *args, **kwargs):
        return GreedyAgglomerativeClusterer().cluster(nx.karate_club_graph(), *args, **kwargs)

    def test_setify_list(self):
        self.assertEqual(set([1, 2]), setify([1, 2]))

    def test_setify_set(self):
        self.assertEqual(set([1, 2, 3]), set([1, 2, 3]))

    def test_setify_mismatch(self):
        self.assertNotEqual(set([1, 2]), setify([1, 2, 3]))

    def test_weight_dict(self):
        self.assertEqual(weight({ 'weight': 4 }), 4.0)

    def test_weight_empty_dict(self):
        self.assertEqual(weight({}), 1.0)

    def test_weight_numeric(self):
        self.assertEqual(weight(6), 6.0)
        self.assertAlmostEqual(weight(7.1), 7.1, 5)

    def test_weighted_edge_count(self):
        self.assertEqual(weighted_edge_count(nx.karate_club_graph()), 156.0)

    def test_weighted_edge_count_weights(self):
        club = nx.karate_club_graph().copy()
        for index, (e1, e2) in enumerate(club.edges()):
            club[e1][e2] = { 'weight': index }
        self.assertEqual(weighted_edge_count(club), 6006.0)

    def test_weighted_edge_count_float_weights(self):
        club = nx.karate_club_graph().copy()
        for index, (e1, e2) in enumerate(club.edges()):
            club[e1][e2] = float(index) + 0.5
        self.assertEqual(weighted_edge_count(club), 6084.0)

    def test_remove_orphans(self):
        club = nx.karate_club_graph().copy()
        club.add_node('test')
        orphans = remove_orphans(club)
        self.assertEqual(orphans, set(['test']))
        self.assertEqual(club.nodes(), nx.karate_club_graph().nodes())

    def test_remove_orphans_fully_connected(self):
        club = nx.karate_club_graph().copy()
        orphans = remove_orphans(club)
        self.assertEqual(orphans, set())
        self.assertEqual(club.nodes(), nx.karate_club_graph().nodes())

    def test_int_map_graph(self):
        orig_club = nx.karate_club_graph()
        club = nx.relabel_nodes(orig_club, { n: str(n) for n in orig_club.nodes_iter() })
        remap = int_graph_mapping(club)
        self.assertEqual(remap.max_node, 33)

        relabled_club = nx.relabel_nodes(club, remap.integer)
        self.assertNotEqual(relabled_club.nodes(), club.nodes())
        # Check that we can map back
        self.assertEqual(sorted(map(lambda n: remap.original[n],
            nx.all_neighbors(relabled_club, remap.integer['33']))),
            sorted(['26', '27', '20', '14', '22', '23', '19', '32', '31', '30', '28',
                '29', '15', '18', '9', '8', '13']))

        repaired_club = nx.relabel_nodes(relabled_club, remap.original)
        self.assertListEqual(sorted(repaired_club.nodes()), sorted(club.nodes()))
        # Check that nothing got reassigned
        self.assertEqual(sorted(nx.all_neighbors(repaired_club, '33')),
            sorted(['26', '27', '20', '14', '22', '23', '19', '32', '31', '30', '28',
                '29', '15', '18', '9', '8', '13']))

    def test_quality_history(self):
        expected_quality = [-0.04980276134122286, -0.03763971071663378, -0.013971071663379343,
            -0.001479289940828389, 0.017751479289940843, 0.04906311637080869,
            0.06122616699539778, 0.0784845496383958, 0.09467455621301776, 0.11111111111111112,
            0.12319197896120973, 0.14587442472057857, 0.1577087442472058, 0.16904996712689022,
            0.1802268244575937, 0.19140368178829717, 0.2081689677843524, 0.21918145956607496,
            0.22937212360289283, 0.24317882971729124, 0.25895792241946086, 0.27925706771860614,
            0.28944773175542404, 0.29930966469428005, 0.3111439842209073, 0.3205128205128205,
            0.32955292570677186, 0.33826429980276135, 0.34935897435897434, 0.3628369493754109,
            0.37598619329388555, 0.38067061143984215, 0.37179487179487175, -5.551115123125783e-17]
        for index, quality in enumerate(self.karate_clustering().quality_history):
            self.assertAlmostEqual(expected_quality[index], quality)

    def test_clustering(self):
        karate = self.karate_clustering()
        self.assertListEqual(karate.clusters(1), [set(range(34))])

        self.assertListEqual(karate.clusters(2),
            [set([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 16, 17, 19, 21]),
             set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31])])

        self.assertListEqual(karate.clusters(3),
            [set([32, 33, 8, 14, 15, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]),
             set([1, 2, 3, 7, 9, 12, 13, 17, 21]),
             set([0, 4, 5, 6, 10, 11, 16, 19])])

        # Requesting more clusters than are available returns the maximum possible instead
        for num_clusters in range(34, 40):
            self.assertListEqual(sorted(karate.clusters(num_clusters), key=lambda c: iter(c).next()),
                [set([index]) for index in range(34)])
        for num_clusters in range(0, -4, -1):
            self.assertListEqual(karate.clusters(num_clusters), [])

    def test_orphans(self):
        graph = nx.Graph();
        graph.add_nodes_from(range(10))
        graph.add_edges_from([[0, 1], [1, 2], [0, 2]])
        dendrogram = GreedyAgglomerativeClusterer().cluster(graph)

        clusters = dendrogram.clusters()
        self.assertIn(set([0, 1, 2]), clusters)
        for node in range(3, 10):
            self.assertIn(set([node]), clusters)

    def test_small_graph(self):
        graph = nx.Graph();
        dendrogram = GreedyAgglomerativeClusterer().cluster(graph)
        self.assertEqual(list(dendrogram.clusters()), [])

        graph.add_node(1)
        dendrogram = GreedyAgglomerativeClusterer().cluster(graph)
        self.assertEqual(list(dendrogram.clusters(1)[0]), [1])
        self.assertEqual(list(dendrogram.clusters()[0]), [1])

    def test_forced_clusters(self):
        forced_clusters = [set([33,0]), set([32,1])]
        karate = self.karate_clustering(forced_clusters=forced_clusters)
        clusters = karate.clusters()

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
