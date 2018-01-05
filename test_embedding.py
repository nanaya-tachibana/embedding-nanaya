import networkx as nx
from embedding import Word2vec, Node2vec

if __name__ == '__main__':
    # model = Word2vec(cbow=0,
    #                  embedding_size=200,
    #                  window=8,
    #                  negative=25,
    #                  learning_rate=0.025,
    #                  linear_learning_rate_decay=1,
    #                  sample=1e-4,
    #                  iters=15)
    # model.train('example_text/text8', n_jobs=4)
    # model.save_word2vec_binary('text8_vec.bin')

    model = Node2vec(cbow=0,
                     embedding_size=200,
                     window=10,
                     negative=5,
                     learning_rate=0.025,
                     linear_learning_rate_decay=1,
                     sample=1e-5,
                     iters=5)
    g = nx.read_edgelist('example_graphs/p2p-Gnutella08.edgelist')
    model.train(g, use_meta_path=1)
    model.save_word2vec_format('p2p_vec.bin')
