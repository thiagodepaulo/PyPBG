from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import time
from tqdm import tqdm
import logging
from scipy.special import logsumexp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


class TPBG(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
              global_max_itr=50, local_threshold = 1e-6, global_threshold = 1e-6,
              save_interval=-1, is_semisupervised=True, feature_names=None):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.local_max_itr = local_max_itr
        self.global_max_itr = global_max_itr
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.is_semisupervised = is_semisupervised
        self.feature_names = feature_names

    def local_propag(self, j):
        local_niter = 0
        words = [x for x in self.X[j].nonzero()[1]]
        log_F = np.log(self.X[j,words].toarray().T)
        log_Aj = self.log_A[j]
        log_Bw = self.log_B[words]
        while local_niter < self.local_max_itr:
            local_niter += 1
            oldA_j = log_Aj
            log_C = log_Aj + log_Bw
            log_C = log_C - logsumexp(log_C, axis=1, keepdims=True)
            log_Aj = np.log(self.alpha + np.sum(np.exp(log_F + log_C),axis=0))
            mean_change = np.mean(abs(log_Aj - oldA_j))
            if mean_change <= self.local_threshold:
                #print('convergiu itr %s' %local_niter)
                break
        self.log_A[j] = log_Aj

    def global_propag(self):
        for i in tqdm(range(self.nwords), ascii=True, desc='global propagation:   '):
            docs = [d for d in self.Xc[:,i].nonzero()[0]]
            log_F = np.log(self.X[docs, i].toarray())
            log_C = self.log_A[docs] + self.log_B[i]
            log_C = log_C - logsumexp(log_C, axis=1, keepdims=True)
            self.log_B[i] = np.log(np.sum(np.exp(log_F + log_C),axis=0))
        self.log_B = self.log_B - logsumexp(self.log_B, axis=0)
        self.log_B = np.log(self.beta + np.exp(self.log_B))

    def _init_matrices(self):
        self.log_A = np.log(np.random.dirichlet(np.ones(self.n_components), self.ndocs))
        self.log_B = np.log(np.random.dirichlet(np.ones(self.nwords), self.n_components).transpose())

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.classes, self.y = np.unique(y, return_inverse=True)
        #if -1 in self.classes: y -= 1
        self.X = X
        self.Xc = X.tocsc()
        self.ndocs, self.nwords = X.shape
        self._init_matrices()
        self.bgp()
        self.components_ = np.exp(self.log_B.T)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not self.X != X:
            return np.exp(self.log_A)
        log_A_back, X_back, local_max_itr_back  = self.log_A, self.X, self.local_max_itr
        ndocs = X.shape[0]
        self.log_A = np.ones((ndocs, self.n_components))
        self.X = X
        self.local_max_itr = 1
        for j in range(ndocs):
            self.local_propag(j)
        log_A_back = self.log_A
        self.log_A, self.X, self.local_max_itr = log_A_back, X_back, local_max_itr_back
        return np.exp(log_A_back)

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        D = self.transform(X)
        nclass = len(self.classes)        
        return self.classes[np.argmax(D[:,:nclass], axis=1)]

    def bgp(self, labelled=None):
        global_niter = 0
        while global_niter < self.global_max_itr :
            self.max = 1
            for j in tqdm(range(self.ndocs), ascii=True, desc=f'docs processed (itr {global_niter})'):
                self.local_propag(j)
                if self.classes[self.y[j]] != -1:
                    self.supress2(j)
            self.global_propag()
            global_niter += 1
            #self.print_top_topics()

    def supress2(self,j):
        cls = self.classes[self.y[j]]
        self.log_A[j].fill(np.log(self.alpha))
        self.log_A[j][cls] = np.max(self.log_A)

    def supress(self,j):
        aux = self.log_A[j][self.y[j]]
        self.log_A[j].fill(np.log(self.alpha))
        self.log_A[j][self.y[j]] = aux


    def print_top_topics(self, n_top_words=10):
        if self.feature_names == None:
            return
        for k, topic in enumerate(self.log_B.transpose()):
            l = [self.feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]
            print(f'topic {k}:' + ', '.join(l))
