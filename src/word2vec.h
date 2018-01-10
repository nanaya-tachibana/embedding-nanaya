#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <pthread.h>

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40
#define EPS 0.00000001


typedef float real;  // Precision of float numbers
long vocab_hash_size;  // Maximum 100 * 0.7 = 70M words in the vocabulary

typedef struct Vocab {
  long long cn;
  // int *point;
  char *word;
  real cumgrad; // cumulative gradient
  // char *code, codelen;
} Vocab;

Vocab *vocab;
int *vocab_hash;
long long vocab_size;

long long *unigram_table;
long long unigram_table_size;
real *syn0;
real *syn1neg;
real *exp_table;
// params
int cbow;
int negative;
long long layer1_size;
int window;
real alpha;
real starting_alpha;
real sample;
long long iter;
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
	       real _init_learning_rate, int _linear_learning_rate_decay,
	       real _sample, int _iter, int _debug_mode, int _n_jobs);
void TrainModel();
void *TrainModelThread(void *id);
