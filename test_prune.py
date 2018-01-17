"""
This test will reproduce the result of one experiment
on the BlogCatalog dataset in the paper
"DeepWalk: Online Learning of Social Representations".
"""
import os

import requests
from zipfile import ZipFile
import shutil
import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import KeyedVectors

from sklearn.model_selection import ShuffleSplit
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

from embedding import Node2vecWithRank, Node2vec


def train_embeddings(g, output):
    model = Node2vec(cbow=0,
                     embedding_size=128,
                     window=10,
                     negative=5,
                     learning_rate=0.025,
                     linear_learning_rate_decay=1,
                     sample=1e-4,
                     iters=5)
    node_names = list(map(str, list(g)))
    model.train(nx.adj_matrix(g),
                node_names,
                path_length=40,
                num_per_vertex=80,
                apply_neu=False)

    model.save_word2vec_format(output)


def run_experiment(X, y, test_size=0.2, random_state=0):
    n = 10
    spliter = ShuffleSplit(n_splits=n,
                           test_size=test_size,
                           random_state=random_state)
    error = 0
    for train_idx, test_idx in spliter.split(X):
        regr = LGBMRegressor()
        train_X = X[train_idx]
        train_y = y[train_idx]
        test_X = X[test_idx]
        test_y = y[test_idx]
        regr.fit(train_X, train_y)

        prediction = regr.predict(test_X)
        prediction[prediction < 0] = 0
        error += mean_squared_error(test_y, prediction) ** 0.5
    return error / n


def download_test_data(filename):
    filename = os.path.join('example_graphs', filename)
    if not os.path.exists(filename):
        print('Download test file.')
        url = 'http://snap.stanford.edu/data/%s.gz' % filename
        zip_file = '.'.join([filename, 'gz'])
        with requests.get(url, stream=True) as r:
            with open(zip_file, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        with ZipFile(zip_file, 'r') as zf:
            zf.extract(filename, path=os.path.dirname(filename))
        os.remove(zip_file)


if __name__ == '__main__':
    download_test_data('cit-HepPh.txt')
    download_test_data('cit-HepPh-dates.txt')

    date_df = pd.read_csv('example_graphs/cit-HepPh-dates.txt',
                          skiprows=1, header=None, sep='\t', dtype=np.str,
                          names=['id', 'date'])
    date_df['id'] = date_df.id.apply(
        lambda x: int(x[2:]) if x.startswith('11') else int(x))
    date_df['date'] = pd.to_datetime(date_df.date)

    cite_df = pd.read_csv('example_graphs/cit-HepPh.txt',
                          skiprows=4, header=None, sep='\t', dtype=np.int,
                          names=['from_id', 'to_id'])
    edges = zip(cite_df.to_id, cite_df.from_id)
    g = nx.Graph(edges)
    paper_before_2000 = date_df[date_df.date < pd.datetime(2000, 1, 1)].copy()

    paper_before_2000['flag'] = paper_before_2000.id.apply(lambda x: x in g)
    paper_before_2000['num'] = 0
    paper_before_2000.loc[paper_before_2000.flag, 'num'] = (
        paper_before_2000.loc[paper_before_2000.flag, 'id'].apply(
            lambda x: len([n for n in g.neighbors(x) if n > 9200000])))

    g_before_2000 = nx.subgraph(g, paper_before_2000.id)
    paper_before_2000['flag'] = paper_before_2000.id.apply(
        lambda x: x in g_before_2000)
    output = 'hepph.bin'
    train_embeddings(g_before_2000, output)
    model = KeyedVectors.load_word2vec_format(output, binary=True)
    X = model.wv[paper_before_2000[paper_before_2000.flag == True].id.apply(str)]
    y = paper_before_2000[paper_before_2000.flag == True].num.values
    error = run_experiment(X, y)
    print(error)
