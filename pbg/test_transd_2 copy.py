#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 19:57:32 2018

@author: thiagodepaulo
"""


from util import SimplePreprocessing
from tpbg_copy_2 import TPBG
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import spacy


### CARREGANDO DADOS  ########################
# categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes'))
# , categories=categories)
newsgroups_test = fetch_20newsgroups(
    subset='test', remove=('headers', 'footers', 'quotes'))
# , categories=categories)
n_train = len(newsgroups_train.data)
n_test = len(newsgroups_test.data)
n = n_train + n_test
all_data = newsgroups_train.data + newsgroups_test.data
print('preprocessing...')
pp = SimplePreprocessing()
M = pp.transform(all_data)
print('done.')
vectorizer = CountVectorizer()
M = vectorizer.fit_transform(M)

# %%
## LABELS ###################################################
indices = np.arange(n)
train_target = newsgroups_train.target
test_target = newsgroups_test.target
all_target = np.concatenate(
    (newsgroups_train.target, newsgroups_test.target), axis=None)

d = {'alt': 'alternative', 'comp': 'computer', 'os': 'operation system', 'ms-windows': 'windows', 'sys': 'system', 'x': 'interface',
     'misc': 'miscellaneous', 'rec': 'recreation', 'autos': 'automobile', 'sci': 'science', 'crypt': 'cryptography', 'med': 'medicine', 'soc': 'society'}
cls_names_ext = [[d.get(s, s) for s in name.split('.')]
                 for name in newsgroups_train.target_names]


print('loading spacy...')
nlp = spacy.load('en_vectors_web_lg')
print('done.')


def get_related_topic(cls_names_ext, topics_list, selected_classes, free_id):
    max_sim = -1
    max = (-1, -1)
    aux1, aux2, aux3 = '', '', ''
    for cls_idx, names in enumerate(cls_names_ext):
        cls_nlp = nlp(' '.join(names))
        if cls_idx in selected_classes:
            continue
        for k, topic in enumerate(topics_list):
            if k not in free_id:
                continue
            topic_nlp = nlp(' '.join(topic))
            sim = topic_nlp.similarity(cls_nlp)
            if sim > max_sim:
                max_sim = sim
                max = (cls_idx, k)
                aux1, aux2, aux3 = cls_nlp, topic_nlp, sim
    print(f"{aux1} -- {aux2} = {aux3}")
    print(max)
    return max


def create_semi_targets(selected_classes, train_target, n_test):
    semi_target = [x if x in selected_classes else -1 for x in train_target]
    semi_target = np.array(semi_target + [-1] * n_test)
    return semi_target


def evaluate(pbg, true_labels, predicted_labels):
    # evaluation
    selected_classes = pbg.get_selected_classes()
    labels_names = ['unknown'] + [newsgroups_train.target_names[cls]
                                  for cls in sorted(selected_classes)]
    labels_id = sorted(set(true_labels))
    print(labels_id)
    print(labels_names)

    cm = confusion_matrix(true_labels, predicted_labels)
    print(classification_report(true_labels, predicted_labels,
                                labels=labels_id, target_names=labels_names))
    print("Confusion matrix")
    print(cm)


# %%
## ITERATION ##########################################
n_class = len(set(train_target))
K = 20
i = 0
pbg = TPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
           local_threshold=1e-6, global_threshold=1e-6,
           feature_names=vectorizer.get_feature_names())

# nonsupervised
semi_target = [-1]*n
print('rodando TPBG...')
pbg.fit(M, semi_target)

# %%
while i < n_class:

    topics_list = pbg.get_topics()
    cls_id, k = get_related_topic(
        cls_names_ext, topics_list, pbg.get_selected_classes(), pbg.free_id)
    pbg.set_class(cls_id, k)
    semi_target = create_semi_targets(
        pbg.get_selected_classes(), train_target, n_test)

    print('rodando TPBG...')
    pbg.fit(M, semi_target)

    # test
    unlabeled_set = np.argwhere(semi_target == -1).ravel()
    predicted_labels = pbg.transduction_[unlabeled_set]
    true_labels = [ul if ul in pbg.get_selected_classes() else -
                   1 for ul in all_target[unlabeled_set]]

    evaluate(pbg, true_labels, predicted_labels)

    pbg.print_top_topics(target_name=newsgroups_train.target_names)
    print(f'fim.{i}')
    i += 1
####
