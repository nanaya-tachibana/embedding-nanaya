cimport cython
from libc.stdlib cimport malloc, free
cimport numpy as np
cimport word2vec

def train_w2v(list vocab,
              np.ndarray[long long, ndim=1, mode='c'] word_freqs,
	      np.ndarray[long long, ndim=1, mode='c'] unigram_table,
              long long unigram_table_size,
	      np.ndarray[real, ndim=1, mode='c'] syn0,
              np.ndarray[real, ndim=1, mode='c'] syn1neg,
              int cbow,
	      long long train_words,
              char *train_file,
              long long embedding_size,
              int negative,
              int window,
              real init_learning_rate,
              int linear_learning_rate_decay,
	      real sample,
              int iters,
              int debug_mode,
              int n_jobs):
    cdef long long i
    cdef char **words
    cdef long long vocab_size,

    vocab_size = len(vocab)
    words = <char **>malloc(vocab_size * cython.sizeof(char_pointer))
    if words is NULL:
        raise MemoryError()
    for i in range(vocab_size):
        words[i] = vocab[i]
    # https://github.com/cython/cython/wiki/tutorials-NumpyPointerToC
    InitModel(words, &word_freqs[0], vocab_size, &unigram_table[0],
              unigram_table_size, &syn0[0], &syn1neg[0], cbow,
              train_words, train_file,
              embedding_size, negative, window, init_learning_rate,
              linear_learning_rate_decay,
              sample, iters, debug_mode, n_jobs)
    with nogil:
        TrainModel()
    free(words)

