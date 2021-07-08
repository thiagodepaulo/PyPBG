from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from tqdm import tqdm
from scipy.special import logsumexp
from numpy import logaddexp


class PBG(BaseEstimator, ClassifierMixin):
    """ Propagation in Bipartite Graph class

    Args:
        BaseEstimator ([type]): [description]
        ClassifierMixin ([type]): [description]
    """

    def __init__(self, n_components, alpha=0.05, beta=0.0001, local_max_itr=50,
                 global_max_itr=50, local_threshold=1e-6, global_threshold=1e-6,
                 save_interval=-1, feature_names=None):
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

    def local_propag(self, j):
        """ Local propagation for a documento d_j

        Args:
            j (integer): index of document d_j
        """
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
        """Global Propagation
        """
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

    def fit(self, X):
        """Fit method

        Args:
            X (sparse matrix): document_X_word sparse matrix

        Returns:
            PBG: self instance of PBG
        """
        self.X = X
        self.Xc = X.tocsc()
        self.ndocs, self.nwords = X.shape
        if not self.is_fitted_:
            self._init_matrices()
        self.bgp()
        self.components_ = np.exp(self.log_B.T)
        self.is_fitted_ = True

        return self

    def transform(self, X):
        """[summary]

        Args:
            X (sparse matrix): documento x word sparse matrix

        Returns:
            [type]: A reduced version of matrix X. This corresponds to document_X_topics matrix
        """
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

    # TODO implement prediction method for nonsupervised procedures of PBG
    def predict(self, X):
        return None

    def bgp(self, labelled=None):
        global_niter = 0
        while global_niter < self.global_max_itr:
            self.max = 1
            for j in tqdm(range(self.ndocs), ascii=True, desc=f'docs processed (itr {global_niter})'):
                self.local_propag(j)
            self.global_propag()
            global_niter += 1
            self.print_top_topics()

    def print_top_topics(self, n_top_words=10, target_name=None):
        """ print to key words for each topic

        Args:
            n_top_words (int, optional): Number of words per topic. Defaults to 10.
            target_name (list of string, optional): List of class names. Defaults to None.

        """
        if self.feature_names is None:
            print('<<feature names not defined>>')
        for k, topic in enumerate(self.log_B.transpose()):
            l_ = [self.feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1: -1]]
            print(f'topic {k}: ' + ', '.join(l_))

    def get_topics(self, n_top_words=10):
        """ Get topics

        Args:
            n_top_words (int, optional): Number of works per topic. Defaults to 10.

        Returns:
            list of list of string: List of list of words, each list is a topic with n_top_words
        """
        l_topics = []
        for topic in self.log_B.transpose():
            l_ = [self.feature_names[i]
                  for i in topic.argsort()[:-n_top_words - 1:-1]]
            l_topics.append(l_)
        return l_topics
