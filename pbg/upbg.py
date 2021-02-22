from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from scipy.special import softmax

class UPBG(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
                 global_max_itr=50, local_threshold=1e-6, global_threshold=1e-6,
                 save_interval=-1, is_semisupervised=True, feature_names=None):
        self.n_components = n_components
        self.alpha = alpha
        self.log_alpha = np.log(alpha)
        self.beta = beta
        self.local_max_itr = local_max_itr
        self.global_max_itr = global_max_itr
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.is_semisupervised = is_semisupervised
        self.feature_names = feature_names
        self.is_fitted_ = False
        self.map_class_ = {-1: -1}  # key=class, value=index position
        self.free_id = set(range(n_components))  # list of index position
        self.map_word_class = {}  # dict of list of class, key=word_id, value=list of classes

    def local_propag(self, j):
        local_niter = 0
        words = [x for x in self.X[j].nonzero()[1]]
        log_F = np.log(self.X[j, words].toarray().T)
        log_Aj = self.log_A[j]
        log_Bw = self.log_B[words]  
        pos_idx = -1 
        if not self.unlabeled[j]:            
            pos_idx = self.map_class_[self.y[j]]
        while local_niter < self.local_max_itr:
            local_niter += 1
            oldA_j = log_Aj
            log_C = log_Aj + log_Bw   
            # supervised dimensions
            log_C = self.local_supress(log_C, pos_idx)
            log_Aj = np.log(self.alpha + np.sum(np.exp(log_F + log_C), axis=0))
            mean_change = np.mean(abs(log_Aj - oldA_j))
            if mean_change <= self.local_threshold:
                #print('convergiu itr %s' %local_niter)
                break
        self.log_A[j] = log_Aj
    
    def local_supress(self, log_C, pos_idx):
        #if not self.unlabeled[j]:        
        sup_log_C, inf_log_C = log_C[:,:self.n_class], log_C[:,self.n_class:] 
        if sup_log_C.shape[0] == 0:
            return log_C         
        if pos_idx != -1:
            # np.max(sup_log_C)
            sup_log_C[:,pos_idx] = 0
        sup_log_C = sup_log_C - logsumexp(sup_log_C, axis=1, keepdims=True)
        inf_log_C = inf_log_C - logsumexp(inf_log_C, axis=1, keepdims=True)
        log_C[:,:self.n_class], log_C[:,self.n_class:] = sup_log_C, inf_log_C
        return log_C

    def local_supress2(self, log_C, pos_idx):
        #if not self.unlabeled[j]:        
        inf_log_C = log_C[:,self.n_class:] 
        nrows = inf_log_C.shape[0]
        if nrows == 0:
            return log_C         
        sup_log_C = np.full((nrows, self.n_class), self.log_alpha )
        if pos_idx != -1:            
            sup_log_C[:,pos_idx] = 0.0        
        inf_log_C = inf_log_C - logsumexp(inf_log_C, axis=1, keepdims=True)
        log_C[:,:self.n_class], log_C[:,self.n_class:] = sup_log_C, inf_log_C
        return log_C

    def global_propag(self):
        for i in tqdm(range(self.nwords), ascii=True, desc='global propagation:   '):
            docs = [d for d in self.Xc[:, i].nonzero()[0]]
            log_F = np.log(self.X[docs, i].toarray())
            log_C = self.log_A[docs] + self.log_B[i]
            log_C = log_C - logsumexp(log_C, axis=1, keepdims=True)            
            self.log_B[i] = logsumexp(log_F + log_C, axis=0)
            #w_cls = self.map_word_class.get(i,-1)
        self.log_B = self.log_B - logsumexp(self.log_B, axis=0)
        self.log_B = np.log(self.beta + np.exp(self.log_B))

    def _init_matrices(self):
        self.log_A = np.log(np.random.dirichlet(
            np.ones(self.n_components), self.ndocs))
        self.log_B = np.log(np.random.dirichlet(
            np.ones(self.nwords), self.n_components).transpose())

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.unlabeled = (y == -1)

        self.n_class = len(np.unique(y))
        self.X = X
        self.y = y
        self.Xc = X.tocsc()
        self.ndocs, self.nwords = X.shape
        if not self.is_fitted_:
            self._init_matrices()
            # create map of
            for cls_id in np.unique(y):
                self.map_class_.setdefault(cls_id, self.free_id.pop())
        self.bgp()
        self.components_ = np.exp(self.log_B.T)
        self.is_fitted_ = True

        # set the transduction item
        # key=idx, value=class_id
        self.inv_map_class_ = {v: k for k, v in self.map_class_.items()}
        self.transduction_ = np.array([self.inv_map_class_.get(idx, -1)
                                       for idx in np.argmax(self.log_A, axis=1)])

        return self

    ## corrigido bug na atualização da matriz A
    def transform(self, X):
        #if not self.X != X:
        #    return np.exp(self.log_A)
        log_A_backup, X_backup, local_max_itr_backup = self.log_A, self.X, self.local_max_itr        
        unlabeled_backup = self.unlabeled
        ndocs = X.shape[0]
        self.log_A = np.full((ndocs, self.n_components), 0.0)
        self.X = X
        self.local_max_itr = 1
        self.unlabeled = np.full(ndocs, True, dtype=bool)
        for j in range(ndocs):
            self.local_propag(j)
        transformed_log_A = self.log_A
        self.log_A, self.X, self.local_max_itr = log_A_backup, X_backup, local_max_itr_backup
        self.unlabeled = unlabeled_backup
        return np.exp(transformed_log_A)

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        D = self.transform(X)
        #nclass = len(self.map_class_)-1
        return np.array([self.inv_map_class_.get(idx, -1)
                         for idx in np.argmax(D, axis=1)])

    def bgp(self, labelled=None):
        global_niter = 0
        while global_niter < self.global_max_itr:            
            for j in tqdm(range(self.ndocs), ascii=True, desc=f'docs processed (itr {global_niter})'):
                self.local_propag(j)
                #if not self.unlabeled[j]:
                #    self.supress3(j)
            self.global_propag() 
            global_niter += 1
            # self.print_top_topics()

    def supress3(self, j):
        #cls = self.classes_[self.y[j]]
        # class id to index position
        #self.log_A = np.log(softmax(self.log_A[j]))
        pos_id = self.map_class_[self.y[j]]
        self.log_A[j][:self.n_class] = np.log(self.alpha)
        self.log_A[j][pos_id] = np.max(self.log_A[j])        
        self.log_A[j][self.n_class:] = np.log(softmax(self.log_A[j][self.n_class:]))

    def supress2(self, j):
        #cls = self.classes_[self.y[j]]
        # class id to index position
        pos_id = self.map_class_[self.y[j]]
        self.log_A[j].fill(np.log(self.alpha))
        self.log_A[j][pos_id] = np.max(self.log_A)

    def supress(self, j):
        aux = self.log_A[j][self.y[j]]
        self.log_A[j].fill(np.log(self.alpha))
        self.log_A[j][self.y[j]] = aux

    def global_supress(self, i, w_cls):
        for cls in w_cls:
            id_pos = self.map_class_.get(cls)        
            self.log_B[i][id_pos] = np.max(self.log_B)

    def print_top_topics(self, n_top_words=10, target_name=None):
        self.inv_map_class_ = {v: k for k, v in self.map_class_.items()}
        if self.feature_names is None:
            return
        for k, topic in enumerate(self.log_B.transpose()):
            l_ = [self.feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1: -1]]
            cls_id = self.inv_map_class_.get(k, -1)
            if cls_id == -1 or target_name is None:
                print(f'topic {k}: ' + ', '.join(l_))
            else:
                print(f'topic {k} [{target_name[cls_id]}]: ' + ', '.join(l_))

    def get_topics(self, n_top_words=10):
        l_topics = []
        for topic in self.log_B.transpose():
            l_ = [self.feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1:-1]]
            l_topics.append(l_)
        return l_topics

    #def get_selected_classes(self):
        #l_ = list(self.map_class_.keys())
        #if -1 in l_:
            #l_.remove(-1)
        #return l_

    def set_class(self, cls_id, pos_id):
        if cls_id not in self.map_class_:
            self.free_id.remove(pos_id)
            self.map_class_[cls_id] = pos_id
