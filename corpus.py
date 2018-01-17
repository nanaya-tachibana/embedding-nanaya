import os
import string
import secrets
import shutil

from sklearn.preprocessing import normalize
import numpy as np
import random_walk


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(secrets.choice(chars) for _ in range(size))


class RandomWalkCorpus:
    """
    Generate random walk corpus.

    g: networkx graph obejct
       The input graph.
    """
    def __init__(self, adj_matrix, node_names, node_types=None):
        self.adj_matrix = adj_matrix
        self.node_names = node_names
        if node_types is None:
            self.node_types = np.zeros(len(node_names), dtype=np.uint8)
        else:
            self.node_types = np.array(node_types, dtype=np.uint8)

        self.outdegree = self.adj_matrix.sum(axis=1, dtype=np.int)
        self.outdegree = np.array(self.outdegree, dtype=np.int).flatten()
        self.indegree = self.adj_matrix.sum(axis=0, dtype=np.int)
        self.indegree = np.array(self.indegree, dtype=np.int).flatten()

    def build_corpus(self,
                     path_length=80,
                     num_per_vertex=10,
                     alpha=0,
                     use_meta_path=0,
                     output_file=None,
                     n_jobs=os.cpu_count()):
        n_types = len(np.unique(self.node_types))

        self.temp_dir = '_'.join(['.random_walk', id_generator()])
        os.mkdir(self.temp_dir)
        output = os.path.join(self.temp_dir, 'random_walk')
        print('Generate random walk corpus.')
        random_walk.build_random_walk_corpus(
            self.adj_matrix,
            self.node_types,
            n_types,
            [name.encode('UTF-8') for name in self.node_names],
            output.encode('UTF-8'),
            path_length, num_per_vertex,
            alpha, use_meta_path, n_jobs)

        temp_files = ['_'.join([output, '%d.txt' % i]) for i in range(n_jobs)]
        if output_file is None:
            output_file = '.'.join([output, 'txt'])
        with open(output_file, 'wb') as dst:
            for temp_file in temp_files:
                with open(temp_file, 'rb') as src:
                    shutil.copyfileobj(src, dst)
                os.remove(temp_file)
        return output_file

    def get_normalized_adj(self, node_list):
        mapping = dict(zip(self.node_names, range(len(self.node_names))))
        node_list = [mapping[node] for node in node_list]
        return normalize(self.adj_matrix[node_list, :], axis=1, norm='l1')

    def __del__(self):
        shutil.rmtree(self.temp_dir)
