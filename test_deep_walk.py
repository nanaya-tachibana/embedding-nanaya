"""
This test will reproduce the result of one experiment
on the BlogCatalog dataset in the paper
"DeepWalk: Online Learning of Social Representations".
"""
import os

import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat
from gensim.models import KeyedVectors

from sklearn.model_selection import ShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MultiLabelBinarizer

from embedding import Node2vec


def train_embeddings(g, output):
    model = Node2vec(cbow=0,
                     embedding_size=128,
                     window=10,
                     negative=5,
                     learning_rate=0.05,
                     linear_learning_rate_decay=1,
                     sample=1e-3,
                     iters=5)
    model.train(g, path_length=40, num_per_vertex=80)
    model.save_word2vec_format(output)


def top_k_rank(prob, k):
    """
    Return the indices of the top k largest prob.
    """
    return np.argpartition(prob, -k)[-k:]


def run_experiment(X, labels, test_size=0.1, random_state=0):
    n = 10
    spliter = ShuffleSplit(n_splits=n,
                           test_size=test_size,
                           random_state=random_state)
    encoder = MultiLabelBinarizer(range(labels.shape[1]))
    micro = 0
    macro = 0
    for train_idx, test_idx in spliter.split(X):
        clf = OneVsRestClassifier(LogisticRegression(C=1),
                                  n_jobs=os.cpu_count())
        train_X = X[train_idx]
        train_y = labels[train_idx]
        test_X = X[test_idx]
        test_y = labels[test_idx]
        clf.fit(train_X, train_y)

        probs = clf.predict_proba(test_X)
        postive_label_counts = np.array(test_y.sum(axis=1),
                                        dtype=np.int).flatten()
        prediction = []
        for prob, k in zip(probs, postive_label_counts):
            prediction.append(top_k_rank(prob, k))
        micro += f1_score(test_y,
                          encoder.fit_transform(prediction),
                          average='micro')
        macro += f1_score(test_y,
                          encoder.fit_transform(prediction),
                          average='macro')
    return micro / n, macro / n


if __name__ == '__main__':
    blogcata = loadmat('example_graphs/blogcatalog.mat')
    g = nx.from_scipy_sparse_matrix(blogcata['network'])
    labels = blogcata['group']

    filename = 'blogcata.bin'
    train_embeddings(g, filename)
    embed = KeyedVectors.load_word2vec_format(filename, binary=True)
    X = embed.wv[list(map(str, range(labels.shape[0])))]
    micro_df = pd.DataFrame(np.zeros(shape=(1, 9)),
                            columns=list(map(lambda x: '{:.0%}'.format(x),
                                             np.linspace(0.1, 0.9, 9))),
                            index=['deepwalk'])
    macro_df = pd.DataFrame(np.zeros(shape=(1, 9)),
                            columns=list(map(lambda x: '{:.0%}'.format(x),
                                             np.linspace(0.1, 0.9, 9))),
                            index=['deepwalk'])
    for i in np.linspace(0.1, 0.9, 9):
        micro, macro = run_experiment(X, labels, test_size=i)
        micro_df.loc['deepwalk', '{:.0%}'.format(1 - i)] = micro
        macro_df.loc['deepwalk', '{:.0%}'.format(1 - i)] = macro
    print('micro score: ')
    print(micro_df)
    print('macro score: ')
    print(macro_df)
