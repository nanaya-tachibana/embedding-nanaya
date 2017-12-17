import gensim, logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


line = gensim.models.word2vec.LineSentence('text8')

model = gensim.models.word2vec.Word2Vec(line, size=200, window=8, min_count=5, workers=4,
                                        negative=25, alpha=0.05, sg=0, iter=15, sample=0.0001)
