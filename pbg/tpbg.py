from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
from scipy.special import logsumexp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from numpy import logaddexp
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from pbg.util import SemiLabelEncoder


class TPBG(BaseEstimator, ClassifierMixin):

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
                 global_max_itr=50, local_threshold=1e-6, global_threshold=1e-6,
                 save_interval=-1, feature_names=None, target_name=None, silence=False,
                 eval_func=None):
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.local_max_itr = local_max_itr
        self.global_max_itr = global_max_itr
        self.local_threshold = local_threshold
        self.global_threshold = global_threshold
        self.feature_names = feature_names
        self.is_fitted_ = False
        self.log_alpha = np.log(self.alpha)
        self.log_beta = np.log(self.beta)
        self.target_name = target_name
        self.silence = silence
        self.eval_func = eval_func
        self.save_interval = save_interval
        self.is_labeled = True

    def local_propag(self, j):
        local_niter = 0
        words = [x for x in self.X[j].nonzero()[1]]
        if len(words) == 0:
            return
        log_F = np.log(self.X[j, words].toarray().T)
        log_Aj = self.log_A[j]
        log_Bw = self.log_B[words]
        log_F_C = None
        while local_niter < self.local_max_itr:
            local_niter += 1
            oldA_j = log_Aj
            log_C = log_Aj + log_Bw
            log_C = log_C - logsumexp(log_C, axis=1, keepdims=True)
            log_F_C = log_F + log_C
            logsumexp_F_C = logsumexp(log_F_C, axis=0, keepdims=True)
            log_Aj = logaddexp(self.log_alpha, logsumexp_F_C)
            mean_change = np.mean(abs(log_Aj - oldA_j))
            if mean_change <= self.local_threshold:
                #print('convergiu itr %s' %local_niter)
                break
        self.log_A[j] = log_Aj
        self.B2[words] += np.exp(log_F_C)

    def global_propag(self):
        sum_columns = np.sum(self.B2, axis=0)
        log_B_norm = np.log(self.B2) - np.log(sum_columns)
        self.log_B = logaddexp(self.log_beta, log_B_norm)
        self.B2 = np.zeros((self.nwords, self.n_components))

    def _init_matrices(self):
        self.log_A = np.log(np.random.dirichlet(
            np.ones(self.n_components), self.ndocs))
        self.log_B = np.log(np.random.dirichlet(
            np.ones(self.nwords), self.n_components).transpose())
        self.B2 = np.zeros((self.nwords, self.n_components))

    def _init_supervised_matrices(self):
        print('oi')
        self._init_matrices()
        for j in range(self.ndocs):
            if not self.unlabeled[j]:
                self.suppress(j)
        for i in tqdm_notebook(range(self.nwords), ascii=True, desc='initialing.[]:   '):
            docs = [d for d in self.X[:, i].nonzero()[0]]
            # if word w_i not belong in train documents set X_train
            if len(docs) == 0:
                self.log_B[i] = np.ones(self.n_components)
                continue
            log_F = np.log(self.X[docs, i].toarray())
            log_A_j = self.log_A[docs]
            log_A_j = log_A_j - logsumexp(log_A_j, axis=1, keepdims=True)
            self.log_B[i] = logsumexp(log_F + log_A_j, axis=0)
        self.log_B = self.log_B - logsumexp(self.log_B, axis=0)
        self.log_B = np.log(self.beta + np.exp(self.log_B))
        self.print_top_topics()

    def create_class_map(self, flat_y):
        for cls_id in np.unique(flat_y):
            if cls_id != -1:  # unlabeled flag
                self.map_class_.setdefault(cls_id, self.free_id.pop())

    def unsupervised_fit(self, X):
        self.is_labeled = False
        self.X = X
        self.ndocs, self.nwords = X.shape
        self.unlabeled = np.empty(self.ndocs)
        self.unlabeled.fill(-1)

        self._init_matrices()
        self.bgp()
        self.components_ = np.exp(self.log_B.T)
        self.is_fitted_ = True

        return self

    def fit(self, X, y, one_class_name=None, one_class_idx=-1):
        X, y = check_X_y(X, y, accept_sparse=True,
                         multi_output=False)
        self.unlabeled = (y == -1)
        self.is_labeled = True

        self.X = X
        #flat_y = [v for l in self.y for v in l] if self.is_multilabel else y
        flat_y = y
        self.sle = SemiLabelEncoder(
            one_class_name=one_class_name, one_class_idx=one_class_idx)
        self.y = self.sle.fit_transform(flat_y)
        self.inv_y = self.sle.inverse_transform(self.y)
        #self.Xc = X.tocsc()
        self.ndocs, self.nwords = X.shape

        if not self.is_fitted_:
            self._init_supervised_matrices()
        self.bgp()
        self.components_ = np.exp(self.log_B.T)
        self.is_fitted_ = True

        self.create_transduction()

        return self

    def create_transduction(self):
        # set the transduction item
        # key=idx, value=class_id
        self.transduction_ = np.array([self.inv_y.get(idx, -1)
                                       for idx in np.argmax(self.log_A, axis=1)])

    def transform(self, X):
        if not self.X != X:
            return np.exp(self.log_A)
        log_A_backup, X_backup, local_max_itr_backup = self.log_A, self.X, self.local_max_itr
        ndocs = X.shape[0]
        self.log_A = np.ones((ndocs, self.n_components))
        self.X = X
        self.local_max_itr = 1
        for j in range(ndocs):
            self.local_propag(j)
        transformed_log_A = self.log_A
        self.log_A, self.X, self.local_max_itr = log_A_backup, X_backup, local_max_itr_backup
        return np.exp(transformed_log_A)

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        D = self.transform(X)
        return self._predict_multiclass(D)

    def _predict_multiclass(self, D):
        return [self.inv_y.get(v, v) for v in np.argmax(D, axis=1)]

    def _predict_multilabel(self, D):
        return D

    def bgp(self, labelled=None):
        global_niter = 0
        while global_niter < self.global_max_itr:
            self.max = 1
            for j in tqdm_notebook(range(self.ndocs), disable=self.silence, ascii=True, desc=f'docs processed (itr {global_niter})'):
                self.local_propag(j)
                if self.is_labeled and not self.unlabeled[j]:
                    self.suppress(j)
            self.global_propag()
            global_niter += 1
            if not self.silence:
                self.print_top_topics()
                if self.eval_func:
                    self.eval_func(self)

    def suppress(self, j):
        pos_id = self.y[j]
        self.log_A[j].fill(np.log(self.alpha))
        self.log_A[j][pos_id] = np.max(self.log_A)

    def print_top_topics(self, n_top_words=10, target_name=None):
        if self.feature_names is None:
            return None
        for k, topic in enumerate(self.log_B.transpose()):
            l_ = [self.feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1: -1]]
            cls_id = self.inv_y.get(k, -1) if self.is_labeled else -1
            cls_name = cls_id if cls_id != -1 else 'None'
            print(f'topic {k} [{ cls_name }] ' + ', '.join(l_))

    def get_topics(self, n_top_words=10):
        l_topics = []
        for topic in self.log_B.transpose():
            l_ = [self.feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1:-1]]
            l_topics.append(l_)
        return l_topics
