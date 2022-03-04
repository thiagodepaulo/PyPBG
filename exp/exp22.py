from pbg import TPBG
import pbg.util
from sklearn.feature_extraction.text import TfidfVectorizer
from random import randint
import numpy as np
import sys
from sklearn.metrics import classification_report
from pbg.util import Params
from sklearn.linear_model import SGDClassifier
from scipy.sparse import vstack


def remove_rows(X, y, idxs):
    if len(idxs) == 0:
        return X, y
    mask = np.ones(X.shape[0], dtype=bool)
    mask[idxs] = False
    w = np.flatnonzero(mask)
    return X[w], y[w]


def most_prob_idx(A):
    A = np.squeeze(A)
    if A.ndim == 2:
        A = np.sum(A, axis=0)
    return np.argmax(A)


def eval_func(model):
    y_predict = model.predict(X_test)
    y_predict = [1 if c == choosed_cls else 0 for c in y_predict]
    y_test2 = [1 if c == choosed_cls else 0 for c in y_test]

    # calcular a métrica
    labels = [0, 1]
    names = ["others", choosed_cls]
    report = classification_report(
        y_test2, y_predict, labels=labels, target_names=names
    )
    print('\n'+report+'\n')

    pos, neg = split_neg_pos(model.log_A, prob_topic_idx, p=1.0)

    clf = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3,
                        random_state=42, max_iter=5, tol=None)
    X_ = vstack((X[pos], X[neg]))
    y_ = np.concatenate((y[pos], y[neg]))
    y_ = [1 if c == choosed_cls else 0 for c in y_]
    clf.fit(X_, y_)
    y_predict = clf.predict(X_test)
    report = classification_report(
        y_test2, y_predict, labels=labels, target_names=names
    )
    print('\n'+report+'\n')
    print(X_.shape)


def non_supervised(args):

    csv_file = args[1]
    n_pos = int(args[2])
    k = int(args[3])
    print('número de tópicos')
    print(k)
    local_itr = float(args[4])
    global_itr = float(args[5])
    alpha = float(args[6])
    beta = float(args[7])

    loader = pbg.util.Loader()
    global X, y
    X, y = loader.load_csv(csv_file, text_column="Text", class_column="Class")
    vect = TfidfVectorizer()
    X = vect.fit_transform(X)

    model = TPBG(
        k,
        alpha=alpha,
        beta=beta,
        local_max_itr=local_itr,
        global_max_itr=global_itr,
        local_threshold=1e-6,
        global_threshold=1e-6,
        save_interval=-1,
        feature_names=vect.get_feature_names_out(),
        silence=False,
    )

    # treinar o modelo
    model.unsupervised_fit(X)
    semi_supervised(X, y, model, n_pos)


def semi_supervised(X, y, model, n_pos):

    global X_test, y_test, choosed_cls

    target_name = list(set(y))
    n_class = len(target_name)
    n_docs = X.shape[0]
    print(n_docs)

    # selecionar aleatoriamente uma classe e n_pos exemplo rotulado
    choosed_cls = target_name[randint(0, n_class - 1)]
    y_choosed_cls = np.where(y == choosed_cls)[0]
    n_pos = min(n_pos, len(y_choosed_cls))
    selected_docs = np.random.choice(y_choosed_cls, size=n_pos, replace=False)

    # most probable topic index
    global prob_topic_idx
    prob_topic_idx = most_prob_idx(model.log_A[selected_docs])
    print(choosed_cls, prob_topic_idx)
    # marcar com -1 todo o restante
    y_train = np.full(n_docs, -1, dtype=object)
    y_train[selected_docs] = choosed_cls

    # insere função de avaliação
    X_test, y_test = remove_rows(X, y, selected_docs)
    model.eval_func = eval_func

    # treinar o modelo
    model.fit(X, y_train, one_class_idx=prob_topic_idx,
              one_class_name=choosed_cls)


def split_neg_pos(A, id_cls, p=0.1):
    r = np.argmax(A, axis=1)  # list of top topics per documents
    m = A[np.arange(A.shape[0]), r]  # list of values
    d = {}
    for doc_id, k in enumerate(r):
        l = d.get(k, [])
        if len(l) == 0:
            d[k] = l
        l.append((doc_id, m[doc_id]))
    n = int(np.sum(r == id_cls) * p)  # p% of documents of class choosed_cls
    pos = [xx[0] for xx in sorted(d[id_cls], key=lambda x: x[1], reverse=True)]
    pos = pos[:n]
    num_dim = len(d.keys()) - 1
    n_resto = max(1, int(n / num_dim))
    neg = [sorted(d[k], key=lambda x: x[1], reverse=True)[:n_resto]
           for k in d.keys() if k != id_cls]
    neg = [x[0] for subx in neg for x in subx]
    return (pos, neg)


if __name__ == "__main__":
    # param = Params(args[0])
    # csv_file = "/home/thiagodepaulo/exp/text-collections/Sequence_of_words_CSV/CSTR.csv"
    # n_pos = 5
    # k = 4
    # local_itr = 10
    # global_itr = 10
    # alpha = 0.05
    # beta = 0.0001
    non_supervised(sys.argv)
