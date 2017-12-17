import networkx as nx
import random_walk


class CorpusBuilder:
    """
    g: networkx graph obejct
    """
    def __init__(self, g):
        self._g = g
        self.node_names = g.nodes()
        mapping = dict(zip(self.node_names, range(len(self.node_names))))
        nx.relabel_nodes(self._g, mapping, copy=False)

    def build_corpus(self,
                     path_length,
                     num_per_vertex,
                     alpha,
                     output='./random_walk',
                     n_jobs=1):
        mapping = dict(zip(self._g.nodes(), range(len(self.node_names))))
        _adj = self._g.adjacency_list()
        adj_list = [_adj[mapping[v]] for v in range(len(self.node_names))]
        random_walk.build_random_walk_corpus(
            adj_list,
            [name.encode('UTF-8') for name in self.node_names],
            output.encode('UTF-8'),
            path_length, num_per_vertex,
            alpha, n_jobs)


if __name__ == '__main__':
    g = nx.read_edgelist('example_graphs/p2p-Gnutella08.edgelist')
    builder = CorpusBuilder(g)
    builder.build_corpus(20, 10, 0, 'data/random_walk', n_jobs=4)
