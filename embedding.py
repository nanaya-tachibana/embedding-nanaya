from array import array
import random
import collections
import numpy as np
import word2vec

import time
from tqdm import *


class Word2vec:

    def __init__(self,
                 cbow=0,
                 embedding_size=100,
                 window=10,
                 negative=5,
                 learning_rate=0.025,
                 sample=1e-5,
                 iters=1,
                 alpha=0.75,
                 debug_mode=2,
                 n_jobs=1,
                 min_count=5,
                 max_unigram_table_size=int(1e8),
                 max_vocab_size=int(1e8)):
        self.trained = False
        self.cbow = cbow
        self.embedding_size = embedding_size

        self.max_unigram_table_size = max_unigram_table_size
        self.unigram_table = np.zeros(max_unigram_table_size, dtype=object)
        self.unigram_table_size = 0
        self.z = 0
        self.alpha = alpha

        self.max_vocab_size = max_vocab_size
        self.word_freqs = collections.defaultdict(int)
        self.word_indices = dict()
        self.syn0 = None
        self.syn1neg = None

        self.window = window
        self.negative = negative
        self.learning_rate = learning_rate
        self.sample = sample
        self.iters = iters
        self.debug_mode = debug_mode
        self.n_jobs = n_jobs
        self.min_count = min_count

    def load(self, filename):
        self.trained = True

    def save(self, filename):
        pass

    def save_embed(self, filename):
        with open(filename, 'wb') as f:
            f.write(b'%d %d\n' % (len(self.word_indices), self.embedding_size))
            words, _ = zip(*sorted(self.word_indices.items(),
                                   key=lambda x: x[1]))
            for i, word in enumerate(words):
                f.write(word.encode('UTF-8'))
                f.write(b' ')
                a = array('f', self.syn0[i, :])
                a.tofile(f)
                f.write(b'\n')

    def train(self, filename):
        train_words = self._load_train_file(filename)
        if not self.trained:
            self.word_freqs['</s>'] += 0
            self.word_indices['</s>'] = 0
        # discard infrequent words
        count = 1
        for word in self.word_freqs:
            freq = self.word_freqs[word]
            if freq < self.min_count:
                train_words -= self.word_freqs[word]
            else:
                self.word_indices[word] = count
                count += 1

        self.unigram_table = np.array(
            [self.word_indices[w]
             for w in self.unigram_table if w in self.word_indices],
            dtype=np.int64)
        self.unigram_table_size = self.unigram_table.shape[0]

        if self.syn1neg is not None:
            temp = self.syn0
            self.syn1neg = np.zeros(self.embedding_size*len(self.word_indices),
                                    dtype=np.float32)
            self.syn1neg[:self.embedding_size*temp.shape[0]] = temp.flatten()
            del temp
        else:
            self.syn1neg = np.zeros(self.embedding_size*len(self.word_indices),
                                    dtype=np.float32)
        if self.syn0 is not None:
            temp = self.syn0
            self.syn0 = np.array(
                np.random.uniform(
                    -0.5, 0.5, self.embedding_size * len(self.word_indices)),
                dtype=np.float32)
            self.syn0[:self.embedding_size*temp.shape[0]] = temp.flatten()
            del temp
        else:
            self.syn0 = np.array(
                np.random.uniform(
                    -0.5, 0.5, self.embedding_size * len(self.word_indices)),
                dtype=np.float32)

        words, _ = zip(*sorted(self.word_indices.items(), key=lambda x: x[1]))
        word_freqs = np.array([self.word_freqs[w] for w in words],
                              dtype=np.int64)

        print("Vocab size: ", len(words))
        print("Words in train file: ", train_words)
        print(word_freqs.shape,
              self.unigram_table.shape, self.unigram_table_size,
              self.syn0.shape, self.syn1neg.shape, self.cbow,
              train_words, filename.encode('UTF-8'),
              self.embedding_size, self.negative, self.window,
              self.learning_rate, self.sample, self.iters,
              self.debug_mode, self.n_jobs)
        word2vec.train_w2v([w.encode('UTF-8') for w in words], word_freqs,
                           self.unigram_table, self.unigram_table_size,
                           self.syn0, self.syn1neg, self.cbow,
                           train_words, filename.encode('UTF-8'),
                           self.embedding_size, self.negative, self.window,
                           self.learning_rate, self.sample, self.iters,
                           self.debug_mode, self.n_jobs)
        self.syn0 = self.syn0.reshape((-1, self.embedding_size))
        self.trained = True

    def _load_train_file(self, filename):
        count = 0
        with open(filename, 'r') as f:
            for line in tqdm(f):
                for word in line.strip('\n ').split(' '):
                    self._add_word(word)
                    count += 1
        return count

    def _add_word(self, word):
        self.word_freqs[word] += 1
        freq = self.word_freqs[word]
        F = freq ** self.alpha - (freq - 1) ** self.alpha
        self.z += F

        if self.unigram_table_size < self.max_unigram_table_size:
            if random.random() <= F:
                self.unigram_table[self.unigram_table_size] = word
                self.unigram_table_size += 1
        else:
            exp = round(self.max_unigram_table_size * F / self.z)
            for _ in range(exp):
                index = random.randint(0, self.max_unigram_table_size - 1)
                self.unigram_table[index] = word


if __name__ == '__main__':
    # model = Word2vec(cbow=1,
    #                  embedding_size=200,
    #                  window=8,
    #                  negative=25,
    #                  learning_rate=0.025,
    #                  sample=1e-4,
    #                  iters=15,
    #                  n_jobs=8)
    # model.train('text8')
    # model.save_embed('embed')
    model = Word2vec(cbow=1,
                     embedding_size=200,
                     window=8,
                     negative=5,
                     learning_rate=0.05,
                     sample=1e-5,
                     iters=3,
                     n_jobs=4)
    model.train('random_walks.csv')
    model.save_embed('wsd')
