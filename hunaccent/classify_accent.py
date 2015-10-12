#-*- coding: utf-8 -*-
from argparse import ArgumentParser
from sys import stdin
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation
import numpy as np
from collections import defaultdict

from corpus_reader import CorpusReader


def run_pipeline(X, y):
    print X.shape, y.shape
    #p = Pipeline([('preproc', StandardScaler()), ('svm', SVC())])
    p = Pipeline([('preproc', StandardScaler()), ('svm', SVC())])
    #p = Pipeline([('pca', PCA(whiten=True)), ('dt', DecisionTreeClassifier())])
    #p = Pipeline([('pca', PCA(whiten=True)), ('logreg', LogisticRegression())])
    print cross_validation.cross_val_score(p, X, y, cv=5)


def convert_labels(labels):
    lab_d = {}
    lab_i = 0
    converted = []
    for l in labels:
        if not l in lab_d:
            lab_d[l] = lab_i
            lab_i += 1
        converted.append(lab_d[l])
    rev_d = {v: k for k, v in lab_d.iteritems()}
    return np.array(converted), rev_d


def parse_args():
    p = ArgumentParser()
    p.add_argument('-w', '--window-size', dest='window', type=int, default=3)
    p.add_argument('--symmetric', action='store_true', help='Use symmetric ngram windows')
    p.add_argument('--accents', type=str, default='áaéeíióoöoőoúuüuűu',
                   help='accent mapping')
    p.add_argument('--filter-punct', action='store_true', default=False)
    p.add_argument('-l', '--lower', action='store_true', default=False)
    p.add_argument('-v', '--verbose', action='store_true', default=False)
    p.add_argument('input_file', nargs='*')
    return p.parse_args()


def main():
    args = parse_args()
    r = CorpusReader(accent_map=args.accents, filter_punct=args.filter_punct, lower=args.lower)
    featdict, labels = r.get_featdict_from_lines(stdin, window=args.window)
    vec = DictVectorizer()
    X = vec.fit_transform(featdict).toarray()
    y, label_d = convert_labels(labels)
    cnt = defaultdict(int)
#    for l in y:
#        cnt[label_d[l]] += 1
#    for k, v in cnt.iteritems():
#        print('{0} {1}'.format(k.encode('utf8'), v))
    #print label_d
    #print(vec.fit_transform(featdict).toarray())
    #print vec.get_feature_names()
    run_pipeline(X, y)


if __name__ == '__main__':
    main()
