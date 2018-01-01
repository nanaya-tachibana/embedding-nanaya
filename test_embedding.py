from embedding import Word2vec

if __name__ == '__main__':
    model = Word2vec(cbow=1,
                     embedding_size=200,
                     window=8,
                     negative=25,
                     learning_rate=0.05,
                     linear_learning_rate_decay=1,
                     sample=1e-4,
                     iters=15,
                     n_jobs=4)
    model.train('example_text/text8')
    model.save_word2vec_binary('text8_vec.bin')
