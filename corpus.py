import os
import shutil
import networkx as nx
import random_walk


class RandomWalkCorpus:
    """
    Generate random walk corpus.

    g: networkx graph obejct
       The input graph.
    """
    def __init__(self, g):
        self._g = g
        self.node_names = g.nodes()
        mapping = dict(zip(self.node_names, range(len(self.node_names))))
        nx.relabel_nodes(self._g, mapping, copy=False)

    def build_corpus(self,
                     path_length=80,
                     num_per_vertex=10,
                     alpha=0,
                     output_dir='.',
                     use_meta_path=0,
                     n_jobs=os.cpu_count()):
        adj_list = [list(self._g.adj[i].keys())
                    for i in range(len(self.node_names))]
        node_types = [self._g.node[i].get('type', 0)
                      for i in range(len(self.node_names))]
        n_types = len(set(node_types))
        node_names = [name.encode('UTF-8') for name in self.node_names]

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        output = os.path.join(output_dir, 'random_walk')
        print('Generate random walk corpus.')
        random_walk.build_random_walk_corpus(
            adj_list,
            node_types,
            n_types,
            node_names,
            output.encode('UTF-8'),
            path_length, num_per_vertex,
            alpha, use_meta_path, n_jobs)

        temp_files = ['_'.join([output, '%d.txt' % i]) for i in range(n_jobs)]
        output_file = '.'.join([output, 'txt'])
        with open(output_file, 'wb') as dst:
            for temp_file in temp_files:
                with open(temp_file, 'rb') as src:
                    shutil.copyfileobj(src, dst)
                os.remove(temp_file)
        return output_file
