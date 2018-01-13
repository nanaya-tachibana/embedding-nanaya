import os
import time

from gensim import logging
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

from embedding import Word2vec


if __name__ == '__main__':
    model = Word2vec(cbow=1,
                     embedding_size=200,
                     window=8,
                     negative=25,
                     learning_rate=0.05,
                     linear_learning_rate_decay=1,
                     sample=1e-4,
                     iters=15)

    now = time.time()
    model.train('example_text/text8', n_jobs=os.cpu_count())
    print('Training compelete. Total time ', time.time() - now)
    model.save_word2vec_format('vec.bin')

    sen = LineSentence('example_text')
