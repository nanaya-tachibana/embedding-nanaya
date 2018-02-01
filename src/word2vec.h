#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <pthread.h>
#include "khash.h"

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define EPS 0.00000001


typedef float real;  // Precision of float numbers


typedef struct Vocab {
  long long freq;
} Vocab;


KHASH_MAP_INIT_STR(VocabHash, long long);

Vocab *vocab;
khash_t(VocabHash) *word_hash;
long long vocab_size;
long long *unigram_table;
long long unigram_table_size;
real *syn0;
real *syn1neg;
real *exp_table;
real *adagrad0;
real *adagrad1neg;
// params
int cbow;
int negative;
long long layer1_size;
int window;
real alpha;
real starting_alpha;
real sample;
long long iter;
real lambda;
int ordered;
int debug_mode;
int linear;

// thread
clock_t start;
long long word_count_actual;
char train_file[MAX_STRING];
long long train_words;
long long file_size;
int num_threads;

void InitModel(char **_words, long long *_word_freqs, long long _vocab_size,
	       long long *_unigram_table, long long _unigram_table_size,
	       real *_syn0, real *_syn1neg, int _cbow,
	       long long _train_words, char *_train_file,
	       long long _embedding_size, int _negative, int _window,
	       real _init_learning_rate, real _sample, int _iter,
	       int _linear_learning_rate_decay, int ordered,
	       int _debug_mode, int _n_jobs);
void TrainModel();
void *TrainModelThread(void *id);
