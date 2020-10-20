
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np
import time
from scipy.special import logsumexp


class PBG(BaseEstimator, TransformerMixin):

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
              global_max_itr=50, local_threshold = 1e-6, global_threshold = 1e-6,
              max_time=18000, save_interval=-1, out_dir='.', out_A='A', out_B='B',
              calc_q=False, debug=False, rand_init=True):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.local_max_itr = local_max_itr
        self.global_max_itr = global_max_itr
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.max_time = max_time    # seconds
        self.save_interval = save_interval
        self.out_dir = out_dir
        self.out_A = out_A
        self.out_B = out_B
        self.calc_q = calc_q
        self.debug = debug
        self.rand_init = rand_init

    def local_propag(self, j):
        local_niter = 0
        words = [x for x in self.X[j].nonzero()[1]]
        F = self.X[j,words].toarray().T
        A_j = self.A[j]
        B_w = self.B[words]
        while local_niter <= self.local_max_itr:
            local_niter += 1
            oldA_j = A_j
            C = A_j * B_w
            C = C/np.sum(C, axis=1, keepdims=True)
            A_j = self.alpha + np.sum((F * C),axis=0)
            #self.A[j] = self.A[j]/np.sum(self.A[j])
            mean_change = np.mean(abs(A_j - oldA_j))
            if mean_change <= self.local_threshold:
                #print('convergiu itr %s' %local_niter)
                break
        self.A[j] = A_j

    def global_propag(self):
        for i in range(self.nwords):
            docs = [d for d in self.Xc[:,i].nonzero()[0]]
            F = self.X[docs, i].toarray()
            C = self.A[docs] * self.B[i]
            C = C/np.sum(C, axis=1, keepdims=True)
            self.B[i] = np.sum((F * C),axis=0)
        self.B = self.B / np.sum(self.B, axis=0)
        self.B += self.beta


    def _init_matrices(self):
        self.A = np.random.dirichlet(np.ones(self.n_components), self.ndocs)
        self.B = np.random.dirichlet(np.ones(self.nwords), self.n_components).transpose()

    def fit(self, X, y=None):
        self.X = X
        self.Xc = X.tocsc()
        self.ndocs, self.nwords = X.shape
        self._init_matrices()
        self.bgp()
        self.components_ = self.B.T

    def transform(self, X):
        if self.X == X:
            return self.A
        A_back, X_back, local_max_itr_back  = self.A, self.X, self.local_max_itr
        ndocs = X.shape[0]
        self.A, self.X, self.local_max_itr = np.ones((ndocs, self.n_components)), X, 0
        for j in range(ndocs):
            local_propag(self, j)
        A_back = self.A
        self.A, self.X, self.local_max_itr = A_back, X_back, local_max_itr_back
        return A_back

    def bgp(self, labelled=None):
        global_niter = 0
        while global_niter <= self.global_max_itr :
            global_niter += 1
            for j in range(self.ndocs):
                #if j % 100 == 0: print(str(j) + " --- " + str(global_niter))
                self.local_propag(j)
            self.global_propag()




class Log_PBG(BaseEstimator, TransformerMixin):

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
              global_max_itr=50, local_threshold = 1e-6, global_threshold = 1e-6,
              max_time=18000, save_interval=-1, out_dir='.', out_A='A', out_B='B',
              calc_q=False, debug=False, rand_init=True):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.local_max_itr = local_max_itr
        self.global_max_itr = global_max_itr
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.max_time = max_time    # seconds
        self.save_interval = save_interval
        self.out_dir = out_dir
        self.out_A = out_A
        self.out_B = out_B
        self.calc_q = calc_q
        self.debug = debug
        self.rand_init = rand_init

    def local_propag(self, j):
        local_niter = 0
        words = [x for x in self.X[j].nonzero()[1]]
        log_F = np.log(self.X[j,words].toarray().T)
        log_Aj = self.log_A[j]
        log_Bw = self.log_B[words]
        while local_niter <= self.local_max_itr:
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
        for i in range(self.nwords):
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

    def fit(self, X, y=None):
        self.X = X
        self.Xc = X.tocsc()
        self.ndocs, self.nwords = X.shape
        self._init_matrices()
        self.bgp()
        self.components_ = np.exp(self.log_B.T)

    def transform(self, X):
        if self.X == X:
            return np.exp(self.log_A)
        log_A_back, X_back, local_max_itr_back  = self.log_A, self.X, self.local_max_itr
        ndocs = X.shape[0]
        self.log_A, self.X, self.local_max_itr = np.ones((ndocs, self.n_components)), X, 0
        for j in range(ndocs):
            local_propag(self, j)
        log_A_back = self.log_A
        self.log_A, self.X, self.local_max_itr = log_A_back, X_back, local_max_itr_back
        return np.exp(log_A_back)

    def bgp(self, labelled=None):
        global_niter = 0
        while global_niter <= self.global_max_itr :
            global_niter += 1
            for j in range(self.ndocs):
                #if j % 100 == 0: print(str(j) + " --- " + str(global_niter))
                self.local_propag(j)
            self.global_propag()
