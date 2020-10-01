#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:57:32 2018

@author: thiagodepaulo
"""
import sys
sys.path.append("..")
import logging
import numpy as np
from tpbg import TPBG
from util import Loader, SimplePreprocessing
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.metrics import f1_score, confusion_matrix, classification_report

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes')
                                       , categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes')
                                       , categories=categories)

n_train = len(newsgroups_train.data)
n_test = len(newsgroups_test.data)
n = n_train + n_test
all_data = newsgroups_train.data + newsgroups_test.data

# labels
indices = np.arange(n)
train_target = newsgroups_train.target
test_target = newsgroups_test.target
all_target = np.concatenate((newsgroups_train.target,newsgroups_test.target),axis=None)
selected_classes = [0,1]
semi_target = [ x if x in selected_classes else -1 for x in train_target]
semi_target = np.array(semi_target + [ -1 ] * n_test)
unlabeled_set = np.argwhere(semi_target == -1).ravel()

print('preprocessing...')
pp = SimplePreprocessing()
M = pp.transform(all_data)
print('done.')

vectorizer = CountVectorizer()
M = vectorizer.fit_transform(M)

print('rodando TPBG...')

K=4
pbg = TPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=2,
            local_threshold = 1e-6, global_threshold = 1e-6, feature_names=vectorizer.get_feature_names())
pbg.fit(M, semi_target)
model = pbg

true_labels = all_target[unlabeled_set]
predicted_labels = pbg.transduction_[unlabeled_set]
print(len(unlabeled_set))
print(set(true_labels))
print(set(predicted_labels))
print(pbg.classes_)
cm = confusion_matrix(true_labels, predicted_labels, labels=pbg.classes_)
print(classification_report(true_labels, predicted_labels))
print("Confusion matrix")
print(cm)



print('fim.')
