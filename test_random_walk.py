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
                     use_meta_path=0,
                     n_jobs=1):
        adj_list = [list(self._g.adj[i].keys())
                    for i in range(len(self.node_names))]
        node_types = [self._g.node[i].get('type', 0)
                      for i in range(len(self.node_names))]
        n_types = len(set(node_types))
        node_names = [name.encode('UTF-8') for name in self.node_names]
        random_walk.build_random_walk_corpus(
            adj_list,
            node_types,
            n_types,
            node_names,
            output.encode('UTF-8'),
            path_length, num_per_vertex,
            alpha, use_meta_path, n_jobs)


if __name__ == '__main__':
    g = nx.read_edgelist('example.txt')
    builder = CorpusBuilder(g)
    builder.build_corpus(20, 10, 0, 'data/random_walk', n_jobs=4)
