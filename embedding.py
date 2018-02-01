import os
from array import array
import random
import heapq
import numpy as np

from sklearn.preprocessing import normalize
from corpus import RandomWalkCorpus
import word2vec

from tqdm import tqdm


class Vocab:

    def __init__(self, max_vocab_size=int(1e8), words=None, word_freqs=None):
        if words is None or word_freqs is None:
            self._vocab = []
            self.word_indices = dict()

            self.min_heap = []
            heapq.heapify(self.min_heap)
            self.space_saving_struct = dict()
        else:
            self._vocab = [{'word': str(word), 'freq': int(freq)}
                           for word, freq in zip(words, word_freqs)]
            self.word_indices = dict(zip(words, range(len(words))))

            self.space_saving_struct = dict()
            for idx in range(len(self)):
                freq = self._vocab[idx]['freq']
                if freq not in self.space_saving_struct:
                    self.space_saving_struct[freq] = {idx}
                else:
                    self.space_saving_struct[freq].add(idx)

            self.min_heap = list(self.space_saving_struct.keys())
            heapq.heapify(self.min_heap)
        self.max_vocab_size = max_vocab_size

    def add_word(self, word):
        """
        Add a word if word is not in the vocab else increase word frequency.
        Parameters
        ----------
        word: str
              The word to be added.

        Returns
        ---------
        idx: int
             The indice of the word.
        """
        if word not in self.word_indices:
            idx = len(self._vocab)
            if idx >= self.max_vocab_size:
                self._space_saving_replace_min(word)
            else:
                self._space_saving_add_word(word)
        else:
            self._space_saving_increase(word)

    def replace_word_by_idx(self, idx, word, reset_freq=True):
        del self.word_indices[self._vocab[idx]['word']]
        self._vocab[idx]['word'] = word

        self.word_indices[word] = idx

    def replace_word(self, original_word, new_word, reset_freq=True):
        self.replace_word_by_idx(self.get_index(original_word),
                                 new_word, reset_freq)

    def _space_saving_find_min(self):
        min_count = self.min_heap[0]
        while not self.space_saving_struct[min_count]:
            heapq.heappop(self.min_heap)
            del self.space_saving_struct[min_count]
            min_count = self.min_heap[0]
        return min_count

    def _space_saving_replace_min(self, word):
        min_count = self._space_saving_find_min()
        idx = self.space_saving_struct[min_count].pop()
        self.replace_word_by_idx(idx, word, reset_freq=False)
        self._vocab[idx]['freq'] = min_count + 1

        if not self.space_saving_struct[min_count]:
            heapq.heappop(self.min_heap)
            del self.space_saving_struct[min_count]
        if min_count + 1 not in self.space_saving_struct:
            heapq.heappush(self.min_heap, min_count + 1)
            self.space_saving_struct[min_count + 1] = set()
        self.space_saving_struct[min_count + 1].add(idx)

    def _space_saving_add_word(self, word):
        idx = len(self._vocab)
        self._vocab.append({'word': word, 'freq': 1})
        self.word_indices[word] = idx

        if 1 not in self.space_saving_struct:  # new value
            heapq.heappush(self.min_heap, 1)
            self.space_saving_struct[1] = set()
        self.space_saving_struct[1].add(idx)

    def _space_saving_increase(self, word):
        idx = self.word_indices[word]
        freq = self._vocab[idx]['freq']
        self._vocab[idx]['freq'] = freq + 1

        self.space_saving_struct[freq].discard(idx)  # move to
        if freq + 1 not in self.space_saving_struct:  # new value
            heapq.heappush(self.min_heap, freq + 1)
            self.space_saving_struct[freq + 1] = set()
        self.space_saving_struct[freq + 1].add(idx)

    def get_frequency(self, word):
        """
        Return the frequency of a word in the vocab.
        Parameters
        ----------
        word: str
              The searching word.

        Returns
        ---------
        freq: int
             The frequency of the word in the vocab.
        """
        idx = self.word_indices.get(word, -1)
        return self._vocab[idx]['freq'] if idx != -1 else 0

    def get_index(self, word):
        """
        Return the index of a word in the vocab.
        Parameters
        ----------
        word: str
              The searching word.

        Returns
        ---------
        idx: int
             The index of the word in the vocab.
             Return -1 if the word is not in the vocab.
        """
        return self.word_indices.get(word, -1)

    def idx(self, i):
        return self._vocab[i]['word']

    def __len__(self):
        return len(self._vocab)

    def __contains__(self, w):
        return w in self.word_indices

    def __iter__(self):
        """
        Sequentially iterator through the vocab.
        """
        return iter(self._vocab)

    def __getitem__(self, w):
        return self.get_frequency(w)

    def __repr__(self):
        return repr(self._vocab)


class Word2vec:

    def __init__(self,
                 cbow=0,
                 embedding_size=100,
                 window=10,
                 negative=5,
                 learning_rate=0.025,
                 linear_learning_rate_decay=1,
                 sample=1e-5,
                 iters=1,
                 alpha=0.75,
                 debug_mode=2,
                 max_unigram_table_size=int(1e8),
                 max_vocab_size=int(1e8)):
        self.trained = False
        self.cbow = cbow
        self.embedding_size = embedding_size

        self.max_unigram_table_size = max_unigram_table_size
        self.unigram_table = np.zeros(max_unigram_table_size, dtype=np.int64)
        self.unigram_table_size = 0
        self.z = 0
        self.alpha = alpha

        self.max_vocab_size = max_vocab_size
        self.vocab = Vocab(max_vocab_size)
        self.syn0 = None
        self.syn1neg = None

        self.window = window
        self.negative = negative
        self.learning_rate = learning_rate
        self.linear_learning_rate_decay = linear_learning_rate_decay
        self.sample = sample
        self.iters = iters
        self.debug_mode = debug_mode

    def load(self, filename):
        self.trained = True
        model = np.load(filename)
        self.syn0 = model['syn0']
        self.syn1neg = model['syn1neg']
        self.z = model['params'][0]
        self.z = float(self.z)
        self.unigram_table_size = int(model['params'][1])
        self.max_unigram_table_size = int(model['params'][2])
        self.vocab = Vocab(max_vocab_size=self.max_vocab_size,
                           words=model['words'].tolist(),
                           word_freqs=model['word_freqs'].tolist())
        self.unigram_table = model['unigram_table']

    def save(self, filename):
        words, word_freqs = zip(*[(v['word'], v['freq']) for v in self.vocab])
        words = np.array(words)
        word_freqs = np.array(word_freqs, dtype=np.uint32)
        np.savez_compressed(filename,
                            words=words,
                            word_freqs=word_freqs,
                            syn0=self.syn0,
                            syn1neg=self.syn1neg,
                            unigram_table=self.unigram_table,
                            params=np.array([self.z, self.unigram_table_size,
                                             self.max_unigram_table_size]))

    def save_word2vec_format(self, filename):
        with open(filename, 'wb') as f:
            f.write(b'%d %d\n' % (len(self.vocab), self.embedding_size))
            for i, v in enumerate(self.vocab):
                word = v['word']
                f.write(word.encode('UTF-8'))
                f.write(b' ')
                a = array('f', self.syn0[i, :])
                a.tofile(f)
                f.write(b'\n')

    def train(self, filename, n_jobs=os.cpu_count()):
        train_words, words, word_freqs = self._setup(filename)
        print('Vocab size: ', len(self.vocab))
        print('Words in train file: ', train_words)
        print('Train embedding model.')
        word2vec.train_w2v([w.encode('UTF-8') for w in words], word_freqs,
                           self.unigram_table, self.unigram_table_size,
                           self.syn0, self.syn1neg, self.cbow,
                           train_words, filename.encode('UTF-8'),
                           self.embedding_size, self.negative, self.window,
                           self.learning_rate, self.sample, self.iters,
                           self.linear_learning_rate_decay,
                           self.debug_mode, n_jobs)
        self.syn0 = self.syn0.reshape((-1, self.embedding_size))
        self.trained = True

    def _setup(self, filename):
        print('Load training data.')
        train_words = self._load_train_file(filename, unigram_online=True)

        vocab_size = len(self.vocab)
        if self.syn1neg is not None:
            temp = self.syn1neg
            self.syn1neg = np.zeros(self.embedding_size * vocab_size,
                                    dtype=np.float32)
            self.syn1neg[:temp.shape[0]] = temp
            del temp
        else:
            self.syn1neg = np.zeros(self.embedding_size * vocab_size,
                                    dtype=np.float32)
        if self.syn0 is not None:
            temp = self.syn0
            self.syn0 = np.array(
                np.random.uniform(
                    -0.5, 0.5, self.embedding_size * vocab_size),
                dtype=np.float32)
            self.syn0[:self.embedding_size*temp.shape[0]] = temp.flatten()
            del temp
        else:
            self.syn0 = np.array(
                np.random.uniform(
                    -0.5, 0.5, self.embedding_size * vocab_size),
                dtype=np.float32)

        words, word_freqs = zip(*[(v['word'], v['freq']) for v in self.vocab])
        word_freqs = np.array(word_freqs, dtype=np.int64)
        return train_words, words, word_freqs

    def _load_train_file(self, filename, unigram_online=False):
        word_count = 0
        with open(filename, 'r') as f:
            for line in tqdm(f):
                for word in line.strip('\n ').split(' '):
                    self.vocab.add_word(word)
                    word_count += 1
                    self._build_unigram_table_online(word)
        return word_count

    def _build_unigram_table_online(self, word):
        freq = self.vocab[word]
        F = freq ** self.alpha - (freq - 1) ** self.alpha
        self.z += F

        idx = self.vocab.get_index(word)
        if self.unigram_table_size < self.max_unigram_table_size:
            if random.random() <= F:
                self.unigram_table[self.unigram_table_size] = idx
                self.unigram_table_size += 1
        else:
            exp = round(self.max_unigram_table_size * F / self.z)
            for _ in range(exp):
                index = random.randint(0, self.max_unigram_table_size - 1)
                self.unigram_table[index] = idx


class Node2vec(Word2vec):

    def train(self,
              adj_matrix,
              node_names,
              node_types=None,
              path_length=80,
              num_per_vertex=10,
              alpha=0,
              meta_path=None,
              output_file=None,
              apply_neu=False,
              n_jobs=os.cpu_count()):
        builder = RandomWalkCorpus(adj_matrix, node_names, node_types)
        corpus_file = builder.build_corpus(
            path_length=path_length,
            num_per_vertex=num_per_vertex,
            alpha=alpha,
            meta_path=meta_path,
            output_file=output_file,
            n_jobs=n_jobs)
        print('')
        super().train(corpus_file, n_jobs=n_jobs)

        if apply_neu:
            node_list = [v['word'] for v in self.vocab]
            A = builder.get_normalized_adj(node_list)
            self.apply_neu(A)
            self.syn0 = normalize(self.syn0)

    def apply_neu(self, A, lambda1=0.5, lambda2=0.25):
        self.syn0[1:, :] = (self.syn0[1:, :] + lambda1 * A.dot(self.syn0[1:, ])
                            + lambda2 * A.dot(A.dot(self.syn0[1:, ])))


# class Node2vecWithRank(Node2vec):

#     def __init__(self,
#                  cbow=0,
#                  embedding_size=100,
#                  window=10,
#                  negative=5,
#                  learning_rate=0.025,
#                  linear_learning_rate_decay=1,
#                  sample=1e-5,
#                  iters=1,
#                  alpha=0.75,
#                  lambd=0,
#                  debug_mode=2,
#                  max_unigram_table_size=int(1e8),
#                  max_vocab_size=int(1e8)):
#         super().__init__(cbow=cbow,
#                          embedding_size=embedding_size,
#                          window=window,
#                          negative=negative,
#                          learning_rate=learning_rate,
#                          linear_learning_rate_decay=linear_learning_rate_decay,
#                          sample=sample,
#                          iters=iters,
#                          alpha=alpha,
#                          debug_mode=debug_mode,
#                          max_unigram_table_size=max_unigram_table_size,
#                          max_vocab_size=max_vocab_size)
#         self.lambd = lambd
#         self.rank1neg = None

#     def train(self,
#               adj_matrix,
#               node_names,
#               node_types=None,
#               path_length=80,
#               num_per_vertex=10,
#               alpha=0,
#               meta_path=None,
#               output_file=None,
#               apply_neu=False,
#               n_jobs=os.cpu_count()):
#         builder = RandomWalkCorpus(adj_matrix, node_names, node_types)
#         corpus_file = builder.build_corpus(
#             path_length=path_length,
#             num_per_vertex=num_per_vertex,
#             alpha=alpha,
#             meta_path=meta_path,
#             output_file=output_file,
#             n_jobs=n_jobs)
#         train_words, words, word_freqs = self._setup(corpus_file)

#         node_list = [v['word'] for v in self.vocab]
#         indices = builder.reindex(node_list)
#         outdegree = builder.outdegree[indices]
#         indegree = builder.indegree[indices]

#         print('')
#         print('Vocab size: ', len(self.vocab))
#         print('Words in train file: ', train_words)
#         print('Train embedding model.')
#         word2vec.train_w2v_with_rank(
#             [w.encode('UTF-8') for w in words], word_freqs,
#             self.unigram_table, self.unigram_table_size,
#             self.syn0, self.syn1neg, self.cbow,
#             train_words, corpus_file.encode('UTF-8'),
#             self.embedding_size, self.negative, self.window,
#             self.learning_rate, self.sample, self.iters,
#             self.linear_learning_rate_decay, outdegree, indegree,
#             self.rank1neg, self.lambd, self.debug_mode, n_jobs)

#         self.syn0 = self.syn0.reshape((-1, self.embedding_size))
#         self.trained = True

#         if apply_neu:
#             node_list = [v['word'] for v in self.vocab if v['word'] != '</s>']
#             A = builder.get_normalized_adj(node_list)
#             self.apply_neu(A)

#     def _setup(self, filename):
#         train_words, words, word_freqs = super()._setup(filename)
#         vocab_size = len(self.vocab)
#         if self.rank1neg is not None:
#             temp = self.rank1neg
#             self.rank1neg = np.zeros(self.embedding_size * vocab_size,
#                                      dtype=np.float32)
#             self.rank1neg[:self.embedding_size*temp.shape[0]] = temp.flatten()
#             del temp
#         else:
#             self.rank1neg = np.zeros(self.embedding_size * vocab_size,
#                                      dtype=np.float32)
#         return train_words, words, word_freqs

#     def apply_neu(self, A, lambda1=0.5, lambda2=0.25):
#         self.syn0[1:, :] = (self.syn0[1:, :] + lambda1 * A * self.syn0[1:, ]
# p                            + lambda2 * A * (A * self.syn0[1:, ]))
