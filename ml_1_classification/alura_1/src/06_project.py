import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
from sklearn.pipeline import Pipeline

SEED = 158
np.random.seed(SEED)
uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"


dados = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
dados.head()


x = dados[["preco", "idade_do_modelo", "km_por_ano"]]
y = dados["vendido"]

treino_x, teste_x, treino_y, teste_y = train_test_split(x, y, test_size=0.25, stratify=y)

print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(treino_x), len(teste_x)))

dummy_stratified = DummyClassifier()
dummy_stratified.fit(treino_x, treino_y)
acuracia = dummy_stratified.score(teste_x, teste_y) * 100
print("Accuracy of the dummy: ", round(acuracia, 2), "%")


modelo = DecisionTreeClassifier(max_depth=3)
modelo.fit(treino_x, treino_y)
acuracia = modelo.score(teste_x, teste_y) * 100
print("Accuracy of the model: ", round(acuracia, 2), "%")


cv = KFold(n_splits=10, shuffle=True)
cv_2 = StratifiedKFold(n_splits=10, shuffle=True)
result = cross_validate(modelo, x, y, cv=cv_2, return_train_score=False)

def print_result(result):
    media = result["test_score"].mean()
    desvio_padrao = result["test_score"].std()
    print("Accuracy of the model with cross validation: [%.2f, %.2f]" % ((media - 2 * desvio_padrao) * 100, (media + 2 * desvio_padrao) * 100))

print_result(result)