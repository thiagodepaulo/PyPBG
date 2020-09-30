#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:57:32 2018

@author: thiagodepaulo
"""
import sys
sys.path.append("..")
import logging
from tpbg import TPBG
from util import Loader, SimplePreprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

#categories = ['alt.atheism', 'talk.religion.misc',
#               'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
                                       #categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
                                       #categories=categories)


print('preprocessing...')
pp = SimplePreprocessing()
M = pp.transform(newsgroups_train.data)
print('done.')

vectorizer = CountVectorizer()
M = vectorizer.fit_transform(M)

print('rodando ...')

pbg = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)
pbg.fit(M, newsgroups_train.target)
model = pbg
#nmf = NMF(n_components=10, random_state=1, solver='mu').fit(M)
#model=nmf

print('fim.')


print('treinando')
M_test = pp.transform(newsgroups_test.data)
M_test = vectorizer.transform(M_test)
y_pre = pbg.predict(M_test)
print(confusion_matrix(y_pre, newsgroups_test.target))
print(f1_score(y_pre, newsgroups_test.target, average='micro'))
print(classification_report(newsgroups_test.target, y_pre, target_names=newsgroups_test.target_names))

#pbg.bgp(M, Mcsc, W, D, A, B)
