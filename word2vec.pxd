cimport numpy as np

ctypedef char * char_pointer
ctypedef float real

cdef extern from 'word2vec.h':
    void InitModel(char **_words, long long *_word_freqs, long long _vocab_size,
	           long long *_unigram_table, long long _unigram_table_size,
	           real *_syn0, real *_syn1neg, int _cbow,
	           long long _train_words, char *_train_file,
	           long long _embedding_size, int _negative, int _window,
                   real _init_learning_rate, real _sample, int _iter,
                   int _linear_learning_rate_decay,
                   int _debug_mode, int _n_jobs)
    void TrainModel() nogil


        
