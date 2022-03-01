from pbg import TPBG
from scipy.io import arff
import pandas as pd
from skmultilearn.dataset import load_from_arff
import pbg.util
from sklearn.feature_extraction.text import TfidfVectorizer
from random import randint
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import sys
from sklearn.metrics import classification_report
from pbg.util import Params


def remove_rows(X, y, idxs):
    mask = np.ones(X.shape[0], dtype=bool)
    mask[idxs] = False
    w = np.flatnonzero(mask)
    return X[w], y[w]


def main():
    args = sys.argv
    # param = Params(args[0])
    # csv_file = "/home/thiagodepaulo/exp/text-collections/Sequence_of_words_CSV/CSTR.csv"
    # n_pos = 5
    # k = 4
    # local_itr = 10
    # global_itr = 10
    # alpha = 0.05
    # beta = 0.0001
    csv_file = args[1]
    n_pos = int(args[2])
    k = int(args[3])
    local_itr = float(args[4])
    global_itr = float(args[5])
    alpha = float(args[6])
    beta = float(args[7])

    loader = pbg.util.Loader()
    X, y = loader.load_csv(csv_file, text_column="Text", class_column="Class")
    target_name = list(set(y))
    n_class = len(target_name)
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

    # selecionar aleatoriamente uma classe e n_pos exemplo rotulado
    choosed_cls = target_name[randint(0, n_class - 1)]
    selected_idx = np.random.choice(
        np.where(y == choosed_cls)[0], size=n_pos, replace=False
    )

    # marcar com -1 todo o restante
    y_train = np.copy(y)
    y_train[[i for i in range(len(y)) if i not in selected_idx]] = -1

    X_test, y_test = remove_rows(X, y, selected_idx)

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

    # insere função de avaliação
    model.eval_func = eval_func

    # treinar o modelo
    model.fit(X, y_train)


if __name__ == "__main__":
    for i in range(10):
        print(f'itr {i} \n')
        main()
