//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#include "word2vec.h"

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
void ReadWord(char *word, FILE *fin, char *eof) {
  int a = 0, ch;
  while (1) {
    ch = getc_unlocked(fin);
    if (ch == EOF) {
      *eof = 1;
      break;
    }
    if (ch == 13) continue;
    if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
      if (a > 0) {
        if (ch == '\n') ungetc(ch, fin);
        break;
      }
      if (ch == '\n') {
        strcpy(word, (char *)"</s>");
        return;
      } else continue;
    }
    word[a] = ch;
    a++;
    if (a >= MAX_STRING - 1) a--;   // Truncate too long words
  }
  word[a] = 0;
}


// Reads a word and returns its index in the vocabulary
int ReadWordIndex(FILE *fin, char *eof) {
  char word[MAX_STRING], eof_l = 0;
  khint_t k;
  ReadWord(word, fin, &eof_l);
  if (eof_l) {
    *eof = 1;
    return -1;
  }
  k = kh_get(VocabHash, word_hash, word);
  if (k == kh_end(word_hash))
    return -1;
  return kh_val(word_hash, k);
}


void InitModel(char **_words, long long *_word_freqs, long long _vocab_size,
	       long long *_unigram_table, long long _unigram_table_size,
	       real *_syn0, real *_syn1neg, int _cbow,
	       long long _train_words, char *_train_file,
	       long long _embedding_size, int _negative, int _window,
	       real _init_learning_rate, int _linear_learning_rate_decay,
	       real _sample, int _iter, long long *_out_degree,
	       long long *_in_degree, real *_rank1neg, float _lambda,
	       int _debug_mode, int _n_jobs) {
  long long i;
  FILE *fin;
  int return_value;
  khint_t k;

  vocab_size = _vocab_size;
  vocab = (Vocab *)malloc(sizeof(Vocab) * vocab_size);
  if (vocab == NULL) {
    fprintf(stderr, "Vocab out of memory\n");
    exit(-1);
  }
  word_hash = kh_init(VocabHash);
  for (i = 0; i < vocab_size; i++) {
    k = kh_put(VocabHash, word_hash, _words[i], &return_value);
    kh_val(word_hash, k) = i;
    vocab[i].freq = _word_freqs[i];
    vocab[i].cumgrad = EPS;
  }

  cbow = _cbow;
  unigram_table = _unigram_table;
  unigram_table_size = _unigram_table_size;
  syn0 = _syn0;
  syn1neg = _syn1neg;
  out_degree = _out_degree;
  in_degree = _in_degree;
  rank1neg = _rank1neg;
  exp_table = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  if (exp_table == NULL) {
    fprintf(stderr, "Exp table out of memory\n");
    exit(-1);
  }
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    exp_table[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);  // Precompute the exp() table
    exp_table[i] = exp_table[i] / (exp_table[i] + 1);  // Precompute f(x) = x / (x + 1)
  }

  negative = _negative;
  layer1_size = _embedding_size;
  window = _window;
  starting_alpha = _init_learning_rate;
  alpha = starting_alpha;
  sample = _sample;
  iter = _iter;
  lambda = _lambda;
  debug_mode = _debug_mode;
  linear = _linear_learning_rate_decay;

  word_count_actual = 0;
  train_words = _train_words;
  strcpy(train_file, _train_file);
  fin = fopen(train_file, "rb");
  fseek(fin, 0, SEEK_END);
  file_size = ftell(fin);
  fclose(fin);
  num_threads = _n_jobs;
}


void TrainModel() {
  long long a;
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  start = clock();
  for (a = 0; a < num_threads; a++)
    pthread_create(&pt[a], NULL, &TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++)
    pthread_join(pt[a], NULL);
  free(pt);
  free(exp_table);
  free(vocab);
}


void *TrainModelThread(void *id) {
  long long a, b, d, cw, word, last_word, sentence_length = 0, sentence_position = 0;
  long long i, j;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label, local_iter = iter;
  unsigned long long next_random = (long long)id;
  char eof = 0;
  real f, g;
  real g1, g2;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cProgress: %.2f%%  Words/thread/sec: %.2fk  ", 13,
         word_count_actual / (real)(iter * train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      if (linear) {
	alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
	if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
      }
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, &eof);
        if (eof) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].freq / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].freq;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (eof || (word_count > train_words / num_threads)) {
      word_count_actual += word_count - last_word_count;
      local_iter--;
      if (local_iter == 0)
	break;
      word_count = 0;
      last_word_count = 0;
      sentence_length = 0;
      fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
      continue;
    }
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    next_random = next_random * (unsigned long long)25214903917 + 11;
    b = next_random % window;
    if (cbow) {  //train the cbow architecture
      // in -> hidden
      cw = 0;
      for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
        c = sentence_position - window + a;
        if (c < 0) continue;
        if (c >= sentence_length) continue;
        last_word = sen[c];
        if (last_word == -1) continue;
        for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
        cw++;
      }
      if (cw) {
        for (c = 0; c < layer1_size; c++) neu1[c] /= cw;
        // NEGATIVE SAMPLING
        if (negative > 0) for (d = 0; d < negative + 1; d++) {
	    if (d == 0) {
	      target = word;
	      label = 1;
	    } else {
	      next_random = next_random * (unsigned long long)25214903917 + 11;
	      target = unigram_table[(next_random >> 16) % unigram_table_size];
	      if (target == 0) target = next_random % (vocab_size - 1) + 1;
	      if (target == word) continue;
	      label = 0;
	    }
	    l2 = target * layer1_size;
	    f = 0;
	    for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
	    if (f > MAX_EXP) g = (label - 1);
	    else if (f < -MAX_EXP) g = (label - 0);
	    else g = (label - exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
	    if (g != 0) {
	      if (linear == 0) {
		vocab[target].cumgrad += pow(g, 2);
		g /= sqrt(vocab[target].cumgrad);
	      }
	      g *= alpha;
	      for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
	      for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
	    }
	  }
        // hidden -> in
        for (a = b; a < window * 2 + 1 - b; a++)
	  if (a != window) {
	    c = sentence_position - window + a;
	    if (c < 0) continue;
	    if (c >= sentence_length) continue;
	    last_word = sen[c];
	    if (last_word == -1) continue;
	    for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
	  }
      }
    } else {  //train skip-gram
      for (a = b; a < window * 2 + 1 - b; a++)
	if (a != window) {
	  c = sentence_position - window + a;
	  if (c < 0) continue;
	  if (c >= sentence_length) continue;
	  last_word = sen[c];

	  if (last_word == -1) continue;
	  l1 = last_word * layer1_size;
	  for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
	  // NEGATIVE SAMPLING
	  if (negative > 0) for (d = 0; d < negative + 1; d++) {
	      if (d == 0) {
		target = word;
		label = 1;
	      } else {
		next_random = next_random * (unsigned long long)25214903917 + 11;
		target = unigram_table[(next_random >> 16) % unigram_table_size];
		if (target == 0) target = next_random % (vocab_size - 1) + 1;
		if (target == word) continue;
		label = 0;
	      }
	      l2 = target * layer1_size;
	      f = 0;
	      for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
	      if (f > MAX_EXP) g = (label - 1);
	      else if (f < -MAX_EXP) g = (label - 0);
	      else g = (label - exp_table[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]);
	      if (g != 0) {
		if (linear == 0) {
		  vocab[target].cumgrad += pow(g, 2);
		  g /= sqrt(vocab[target].cumgrad);
		}
		g *= alpha;
		for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
		for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
	      }
	    }
	  // Learn weights input -> hidden
	  for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
	}
    }
    if (rank1neg != NULL && sentence_position != 0) {
      i = sen[sentence_position - 1];
      l1 = i * layer1_size;
      j = word;
      l2 = j * layer1_size;
      g = 0;
      for (c = 0; c < layer1_size; c++) g += syn0[c + l1] * rank1neg[c + l1];
      g = out_degree[i] != 0 ? g / out_degree[i] * (g > 0) : 0;
      f = 0;
      for (c = 0; c < layer1_size; c++) f += syn0[c + l2] * rank1neg[c + l2];
      f = in_degree[j] != 0 ? f / in_degree[j] * (f > 0) : 0;
      g1 = out_degree[i] != 0 ? alpha * lambda * 2 * in_degree[j] / out_degree[i] * (f - g) * (g > 0) : 0;
      g2 = -alpha * lambda * 2 * (f - g) * (f > 0);
      if (g1 != 0)
	for (c = 0; c < layer1_size; c++) {
	  syn0[c + l1] += g1 * rank1neg[c + l1];
	  rank1neg[c + l1] += g1 * syn0[c + l1];
	}
      if (g2 != 0)
	for (c = 0; c < layer1_size; c++) {
	  syn0[c + l2] += g2 * rank1neg[c + l2];
	  rank1neg[c + l2] += g2 * syn0[c + l2];
	}
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}


int main() {
  return 0;
}
