import numpy as np
import pandas as pd
import csv
from sklearn.calibration import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt
from datetime import datetime as dt
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz


def read_csv(file_path, delimiter=';'):
    with open(file_path, 'r') as f:
        yield from [row for row in csv.reader(f, delimiter=delimiter)]

# passing header columns reading csv in pandas


if __name__ == '__main__':

    uri = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
    df_data = pd.read_csv(uri, header=0, names=['milhas_por_ano', 'ano_do_modelo', 'preco', 'vendido'])

    df_data.vendido = df_data.vendido.map({"yes": 1, "no": 0})
    ano_atual = dt.today().year
    df_data["idade_do_modelo"] = ano_atual - df_data.ano_do_modelo
    df_data["km_por_ano"] = df_data.milhas_por_ano * 1.60934
    df_data = df_data.drop(columns=["milhas_por_ano", "ano_do_modelo"], axis=1)
    print(df_data.head())

    x = df_data[["preco", "idade_do_modelo", "km_por_ano"]]
    y = df_data["vendido"]

    SEED = 5
    np.random.seed(SEED)
    
    raw_train_x, raw_test_x, train_y, test_y = train_test_split(x, y, random_state=SEED, test_size=0.25, stratify=y)
    print("Treinaremos com %d elementos e testaremos com %d elementos" % (len(raw_train_x), len(raw_test_x)))

    # Scaling features are not necessary for Decision Tree
    # scaler = StandardScaler()
    # scaler.fit(raw_train_x)
    # train_x = scaler.transform(raw_train_x)
    # test_x = scaler.transform(raw_test_x)


    modelo = DecisionTreeClassifier(max_depth=2)
    modelo.fit(raw_train_x, train_y)
    predict = modelo.predict(raw_test_x)
    accuracy = accuracy_score(test_y, predict)
    print("Accuracy of the model: ", round(accuracy * 100, 2), "%")


    # Baseline
    dummy = DummyClassifier()
    dummy.fit(raw_train_x, train_y)
    predict_dummy = dummy.score(raw_test_x, test_y)
    print("Accuracy of the dummy: ", round(predict_dummy * 100, 2), "%")

    dot_data = export_graphviz(modelo, out_file=None, feature_names=x.columns, filled=True, rounded=True, class_names=["não", "sim"])
    grafico = graphviz.Source(dot_data)
    grafico.view()


