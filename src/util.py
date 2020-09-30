#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:46:22 2017

@author: thiagodepaulo
"""
import re
import glob
import os.path
import codecs
import numpy as np
from collections import defaultdict
import re
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer


class Loader:

    def __init__(self):
        pass

    # load supervised dataset
    def from_files(self, path, encod="ISO-8859-1"):
        dirs = glob.glob(os.path.join(path,'*'));
        class_names = []
        class_idx = []
        cid = -1
        corpus = []
        for _dir in dirs:
            cid+= 1
            class_names.append(os.path.basename(_dir))
            arqs = glob.glob(os.path.join(_dir,'*'))
            for arq in arqs:
                with codecs.open(arq, "r", encod) as myfile:
                    data=myfile.read().replace('\n', '')
                corpus.append(data)
                class_idx.append(cid)
        result = {'corpus':corpus, 'class_index': class_idx, 'class_names':class_names}
        return result


    def from_files_2(self, path, encod="UTF-8"):
        corpus = []
        for arq in glob.iglob(path):
            with codecs.open(arq, "r", encod) as myfile:
                corpus.append(myfile.read().replace('\n',''))
        return corpus


    def from_text_line_by_line(self, arq):
        doc = []
        for line in open(arq):
            doc.append(line)
        return doc

    def _str_to_list(self, s):
        _s = re.split(',|{|}',s)
        return [ x for x in _s if len(x) > 0]

    def _str_to_date(self, s):
        pass

    def _convert(self, x, i, attr_list):
        if attr_list[i][1] == self.attr_numeric[1]:
            return float(x)
        elif attr_list[i][1] == self.attr_numeric[2]:
            return int(x)
        elif attr_list[i][1] == self.attr_string[0]:
            return x.replace("'","").replace('\'',"").replace('\"',"")
        else:
            return x.replace("'","").replace('\'',"").replace('\"',"")


    def from_arff(self, arq, delimiter=','):
        relation_name = ''
        attr_count = 0
        attr_list = []
        data = []
        self.attr_numeric = ['numeric', 'real', 'integer']
        self.attr_string = ['string']
        self.attr_date = ['date']
        read_data = False
        for line in open(arq):
            line = line.lower().strip()
            if line.startswith('#'): continue
            if read_data:
                vdata = line.split(delimiter)
                data.append([ self._convert(x,i,attr_list) for i,x in enumerate(vdata) ])
            elif not line.startswith('#'):
                if line.startswith('@relation'):
                    relation_name = line.split()[1]
                elif line.startswith('@attribute'):
                    attr_count += 1
                    attr = line.split()
                    attr_type = attr[2]
                    if attr_type in self.attr_numeric or attr_type in self.attr_string:
                        attr_list.append((attr[1], attr[2]))
                    elif attr_type in self.attr_date:
                        attr_list.append((attr[1], self._str_to_date(attr[2])))
                    else:
                        attr_list.append((attr[1], self._str_to_list(''.join(attr[2:]))))
                elif line.startswith('@data'):
                    read_data = True
                    continue
        d = dict()
        d['attributes'] = attr_list
        d['data'] = data
        d['relation'] = relation_name
        return d

    def from_sparse_arff(self,arq, delimiter=','):
        pass


class ConfigLabels:

    def __init__(self, unlabelled_idx=-1, list_n_labels=[10,20,30,40,50]):
        self.unlabelled_idx = unlabelled_idx
        self.list_n_labels = list_n_labels

    def pick_n_labelled(self, y, n_labelled_per_class):
        class_idx = set(y)
        labelled_idx = []
        for c in class_idx:
            r=np.isin(y, c)
            labelled_idx = np.concatenate((labelled_idx, np.random.choice(np.where(r)[0], n_labelled_per_class)))
        return labelled_idx.astype(int)

    # colocar o valor self.unlabelled_idx nos exemplos não rotulados de y
    def config_labels(self, y,labelled):
        unlabelled = []
        for i in range(len(y)):
            if i not in labelled:
                y[i] = self.unlabelled_idx
                unlabelled.append(i)
        return unlabelled

    # return a dictionary key=<number of labels>, value is a list: [vector
    # with unlabels and labels, vector only with unlabels]
    def select_labelled_index(self, y, n_labels=[10,20,30,40,50]):
        dict_y = {}
        #pega ni documentos rotulados por classe
        for ni in n_labels:
            dict_y[ni] = [np.array(y), None]
            nl = self.pick_n_labelled(y,ni)
            unl = self.config_labels(dict_y[ni][0],nl)
            dict_y[ni][1] = unl
        return dict_y

    def fit(self, y):
        dict_y = self.select_labelled_index(y, n_labels=self.list_n_labels)
        self.unlabelled_idx = {k:dict_y[k][1] for k in dict_y}
        self.semi_labels = { k:dict_y[k][0] for k in dict_y}
        return self


class RandMatrices:

    def create_rand_maps(self, D, W, K):
        A = self.create_rand_matrix_A(D, K)
        B = self.create_rand_matrix_B(W, K)
        Amap = dict()
        Bmap = dict()
        for j, d_j in enumerate(D):
            Amap[d_j] = A[j]
        for i, w_i in enumerate(W):
            Bmap[w_i] = B[i]
        return Amap, Bmap

    def oi(self):
        print('oi')

    def create_rand_matrices(self, D, W, K):
        return (self.create_rand_matrix_A(D, K), self.create_rand_matrix_B(W, K))

    def create_rand_matrix_B(self, W, K):
        N = len(W)     # number of words
        return np.random.dirichlet(np.ones(N), K).transpose()     # B (N x K) matrix

    def create_rand_matrix_A(self, D, K):
        M = len(D)    # number of documents
        return np.random.dirichlet(np.ones(K), M)    # A (M x K) matrix

    def create_ones_matrix_A(self, D, K):
        M = len(D)    # number of documents
        return np.ones(shape=(M,K))

    def create_label_init_matrix_B(self, M, D, y, K, beta=0.0, unlabelled_idx=-1):
        ndocs,nwords = M.shape
        B = np.full((nwords, K),beta)
        count={}
        for word in range(nwords): count[word] = defaultdict(int)
        rows,cols = M.nonzero()
        for row,col in zip(rows,cols):
            label = y[row]
            if label != unlabelled_idx:
                count[col][y[row]] += M[row,col]
                count[col][-1] += M[row,col]
        for word in range(nwords):
            for cls in count[word]:
                if cls != -1: B[word][cls] = (beta + count[word][cls])/(beta + count[word][-1])
        return B

    def create_label_init_matrices(self, X, D, W, K, y, beta=0.0, unlabelled_idx=-1):
        return (self.create_rand_matrix_A(D, K), self.create_label_init_matrix_B(X, D, y, K, beta, unlabelled_idx))

    def create_fromB_matrix_A(self, X, D, B):
        K = len(B[0])
        M = len(D)    # number of documents
        A = np.zeros(shape=(M,K))
        for d_j in D:
            for w_i, f_ji in zip(X.indices[X.indptr[d_j]:X.indptr[d_j+1]],
                       X.data[X.indptr[d_j]:X.indptr[d_j+1]]):
                A[d_j] += f_ji * B[w_i]

        return A

    def create_fromA_matrix_B(self, A):
        K = len(A[0])
        N = self.G.b_len()     # number of words
        B = np.zeros(shape=(N,K))
        for w_i in self.G.b_vertices():
                for d_j, f_ji in self.G.w_b_neig(w_i):
                        B[w_i] += f_ji * A[d_j]
        return self.normalizebycolumn(B)


class SimplePreprocessing():

    def __init__(self):
        pass

    def transform(self, docs):
        tokenizer = RegexpTokenizer(r'\w+')
        stopwords = nltk.corpus.stopwords.words('english') + ['would']
        pattern = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')
        for idx in range(len(docs)):
            docs[idx] = docs[idx].lower()  # Convert to lowercase.
            docs[idx] = pattern.sub('',docs[idx])  # remove stopwords
            docs[idx] = re.sub(r'[^a-z]',' ',docs[idx]) # remove non-alphabet characters
            docs[idx] = tokenizer.tokenize(docs[idx])  # Split into words.

        # Remove numbers, but not words that contain numbers.
        #docs = [[token for token in doc if not token.isdigit()] for doc in docs]

        # Remove words that are only one character.
        docs = [[token for token in doc if len(token) > 3] for doc in docs]

        # Lemmatize all words in documents.
        lemmatizer = WordNetLemmatizer()
        docs = [[lemmatizer.lemmatize(token) for token in doc] for doc in docs]
        docs = [ ' '.join(doc) for doc in docs]
        return docs




#
#l = Loader()
##d = l.from_arff('datasets/SyskillWebert.arff')
#d = l.from_files('/exp/datasets/docs_rotulados/SyskillWebert-Parsed')
#
#import preprocessor
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.naive_bayes import MultinomialNB
#from sklearn.linear_model import SGDClassifier
#from sklearn.pipeline import Pipeline
#import numpy as np
#from sklearn import metrics
#from sklearn.model_selection import GridSearchCV
#
##text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
##                     ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),])
#
##parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3),}
#
#text_clf = Pipeline([('text_preproc',preprocessor.Preprocessor()), ('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
#                     ('clf', MultinomialNB()),])
#
#parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False),}
#
#
#gs_clf = GridSearchCV(text_clf, parameters,  cv=10, n_jobs=-1)
#gs_clf = gs_clf.fit(d['corpus'], d['class_index'])
#print(gs_clf.cv_results_)
#
