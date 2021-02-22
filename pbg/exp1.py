from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from util import SimplePreprocessing
from zpbg import ZPBG
from tpbg import TPBG
from upbg import UPBG
import numpy as np


### CARREGANDO DADOS  ########################
categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(
    subset='train', remove=('headers', 'footers', 'quotes')
 , categories=categories)
newsgroups_test = fetch_20newsgroups(
    subset='test', remove=('headers', 'footers', 'quotes')
 , categories=categories)

print('preprocessing...')
pp = SimplePreprocessing()
M_train = pp.transform(newsgroups_train.data)
M_test = pp.transform(newsgroups_test.data)

print('done.')
vectorizer = TfidfVectorizer() #ngram_range=(1, 3)
M_train = vectorizer.fit_transform(M_train)
M_test = vectorizer.transform(M_test)

d = {'alt': 'alternative', 'comp': 'computer', 'os': 'operation system', 'ms-windows': 'windows', 'sys': 'system', 'x': 'interface',
     'misc': 'miscellaneous', 'rec': 'recreation', 'autos': 'automobile', 'sci': 'science', 'crypt': 'cryptography', 'med': 'medicine', 'soc': 'society'}
cls_names_ext = [[d.get(s, s) for s in name.split('.')]
                 for name in newsgroups_train.target_names]     

print(cls_names_ext)


### PBG
n_class = len(set(newsgroups_train.target))
print(f'nclass {n_class}')
K=100
pbg = UPBG(K, alpha=0.005, beta=0.001, local_max_itr=50, global_max_itr=10,
           local_threshold=1e-6, global_threshold=1e-6,
           feature_names=vectorizer.get_feature_names())

#pbg = MultinomialNB(alpha=0.1)
#pbg = SVC()

print("fitting...")
pbg.fit(M_train, newsgroups_train.target)
print('done')

pbg.print_top_topics(n_top_words=5, target_name=newsgroups_train.target_names)

y_pred = pbg.predict(M_test)

score = f1_score(y_pred, newsgroups_test.target, average='micro')
print(score)



