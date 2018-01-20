import networkx as nx
from corpus import RandomWalkCorpus


if __name__ == '__main__':
    g = nx.read_edgelist('test_edgelist')
    types = [0, 1, 1, 2, 3, 0]
    builder = RandomWalkCorpus(nx.adj_matrix(g), list(g), types)
    builder.build_corpus(10, 1, output_file='p2p.txt', meta_path=[0,1,2,3], n_jobs=4)
