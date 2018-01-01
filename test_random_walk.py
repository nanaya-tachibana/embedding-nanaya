import networkx as nx
from corpus import RandomWalkCorpus


if __name__ == '__main__':
    g = nx.read_edgelist('example_graphs/p2p-Gnutella08.edgelist')
    builder = RandomWalkCorpus(g)
    builder.build_corpus(80, 10, 0, '.', n_jobs=4)
