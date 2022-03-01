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

csv_file = "/home/thiagodepaulo/exp/text-collections/Sequence_of_words_CSV/CSTR.csv"
n_class = 4
n_pos = 5
k = 4

loader = pbg.util.Loader()
X, y = loader.load_csv(csv_file, text_column="Text", class_column="Class")
target_name = list(set(y))
vect = TfidfVectorizer()
X = vect.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

model = TPBG(
    k,
    alpha=0.05,
    beta=0.0001,
    local_max_itr=10,
    global_max_itr=10,
    local_threshold=1e-6,
    global_threshold=1e-6,
    save_interval=-1,
    feature_names=vect.get_feature_names_out(),
    silence=False,
)

# selecionar aleatoriamente uma classe e n_pos exemplo rotulado
choosed_cls = list(set(y_train))[randint(0, n_class - 1)]
selected_idx = np.random.choice(
    np.where(y_train == choosed_cls)[0], size=n_pos, replace=False
)

# marcar com -1 todo o restante
y_train[[i for i in range(len(y_train)) if i not in selected_idx]] = -1

# treinar o modelo
model.fit(X_train, y_train)

# classificar os exemplos de teste
y_predict = model.predict(X_test)
y_predict2 = [1 if c == choosed_cls else 0 for c in y_predict]
y_test2 = [1 if c == choosed_cls else 0 for c in y_test]

# calcular a métrica
print(confusion_matrix(y_test2, y_predict2))
print(f1_score(y_test2, y_predict2, average="weighted"))
print(f1_score(y_test2, y_predict2, average="micro"))
print(f1_score(y_test2, y_predict2, average="macro"))

# repita n vezes
