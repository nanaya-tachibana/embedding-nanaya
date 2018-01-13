import os
import time
import shutil
from zipfile import ZipFile

import requests
from gensim import logging
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.word2vec import LineSentence

from embedding import Word2vec


if __name__ == '__main__':
    filename = 'example_text/text8'
    if not os.path.exists(filename):
        print('Download test file.')
        url = 'http://mattmahoney.net/dc/text8.zip'
        zip_file = '.'.join([filename, 'zip'])
        with requests.get(url, stream=True) as r:
            with open(zip_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        with ZipFile(zip_file, 'r') as zf:
            zf.extract('text8', path=os.path.dirname(filename))
        os.rm(zip_file)

    model = Word2vec(cbow=1,
                     embedding_size=200,
                     window=8,
                     negative=25,
                     learning_rate=0.05,
                     linear_learning_rate_decay=1,
                     sample=1e-4,
                     iters=15)

    now = time.time()
    model.train(filename, n_jobs=os.cpu_count())
    print('Training compelete. Total time ', time.time() - now)
    model.save_word2vec_format('vec.bin')

    sen = LineSentence('example_text')
