from array import array
import random
import numpy as np
import word2vec

from tqdm import tqdm


class Vocab:

    def __init__(self):
        self._vocab = []
        self.word_indices = dict()

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
            self.word_indices[word] = idx
            self._vocab.append({'word': word, 'freq': 1})
        else:
            idx = self.word_indices[word]
            self._vocab[idx]['freq'] += 1
        return idx

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

    def get_indice(self, word):
        """
        Return the indice of a word in the vocab.
        Parameters
        ----------
        word: str
              The searching word.

        Returns
        ---------
        idx: int
             The indice of the word in the vocab.
             Return -1 if the word is not in the vocab.
        """
        return self.word_indices.get(word, -1)

    def filtering(self, min_count):
        """
        Remove the words with frequency less than min_count.
        Parameters
        ----------
        min_count: int
                   The minimum count.

        Returns
        ---------
        count: int
               The sum of frequency of all the removed words.
        """
        count = 0
        vocab = []
        for d in self:
            if d['freq'] < min_count:
                count += d['freq']
            else:
                vocab.append(d)
        self._vocab = vocab
        self.word_indices = {d['word']: i for i, d in enumerate(self)}
        return count

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

        self.vocab = Vocab()
        self.syn0 = None
        self.syn1neg = None

        self.window = window
        self.negative = negative
        self.learning_rate = learning_rate
        self.linear_learning_rate_decay = linear_learning_rate_decay
        self.sample = sample
        self.iters = iters
        self.debug_mode = debug_mode
        self.n_jobs = n_jobs
        self.min_count = min_count

    def load(self, filename):
        self.trained = True

    def save(self, filename):
        pass

    def save_word2vec_binary(self, filename):
        with open(filename, 'wb') as f:
            f.write(b'%d %d\n' % (len(self.vocab), self.embedding_size))
            for i, v in enumerate(self.vocab):
                word = v['word']
                f.write(word.encode('UTF-8'))
                f.write(b' ')
                a = array('f', self.syn0[i, :])
                a.tofile(f)
                f.write(b'\n')

    def train(self, filename):
        if not self.trained:
            train_words = self._load_train_file(filename)
        else:
            train_words = self._load_train_file(filename, unigram_online=True)
        self.vocab.add_word('</s>')

        vocab_size = len(self.vocab)
        if self.syn1neg is not None:
            temp = self.syn0
            self.syn1neg = np.zeros(self.embedding_size * vocab_size,
                                    dtype=np.float32)
            self.syn1neg[:self.embedding_size*temp.shape[0]] = temp.flatten()
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
                           self.learning_rate, self.linear_learning_rate_decay,
                           self.sample, self.iters,
                           self.debug_mode, self.n_jobs)
        self.syn0 = self.syn0.reshape((-1, self.embedding_size))
        self.trained = True

    def _load_train_file(self, filename, unigram_online=False):
        word_count = 0
        with open(filename, 'r') as f:
            for line in tqdm(f):
                for word in line.strip('\n ').split(' '):
                    self.vocab.add_word(word)
                    word_count += 1
                    if unigram_online:
                        self._build_unigram_table_online(word)

        word_count -= self.vocab.filtering(self.min_count)
        if not unigram_online:
            self._build_unigram_table()
        else:
            self.unigram_table = np.array(
                [self.vocab.get_indice[w]
                 for w in self.unigram_table if w in self.vocab],
                dtype=np.int64)
        return word_count

    def _build_unigram_table(self):
        self.unigram_table = np.zeros(self.max_unigram_table_size,
                                      dtype=np.int64)
        self.unigram_table_size = self.max_unigram_table_size
        words_pow = sum(v['freq'] ** self.alpha for v in self.vocab)
        idx = 0
        cumulated = 0
        for v in self.vocab:
            word, freq = v['word'], v['freq']
            cumulated += freq ** self.alpha / words_pow
            while (idx < self.unigram_table_size
                   and idx / self.max_unigram_table_size <= cumulated):
                self.unigram_table[idx] = self.vocab.get_indice(word)
                idx += 1
        self.z = words_pow

    def _build_unigram_table_online(self, word):
        freq = self.vocab[word]
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
