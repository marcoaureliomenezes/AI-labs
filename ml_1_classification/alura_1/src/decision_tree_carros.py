from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import graphviz
from datetime import datetime as dt


def wrangle_data(df):
    df.vendido = df.vendido.map({"yes": 1, "no": 0})
    ano_atual = dt.today().year
    df["idade_do_modelo"] = ano_atual - df.ano_do_modelo
    df["km_por_ano"] = df.milhas_por_ano * 1.60934
    df = df.drop(columns=["milhas_por_ano", "ano_do_modelo"], axis=1)
    return df

def select_features_and_train(df_handled_data):
    df_X = df_handled_data[["preco", "idade_do_modelo", "km_por_ano"]]
    df_Y = df_handled_data["vendido"]
    return train_test_split(df_X, df_Y, test_size=0.25, stratify=df_Y)


def model_score(model, X_test, Y_test):
    return round(model.score(Y_test, X_test) * 100, 2)

def baseline_dummy(X_train, X_test, Y_train, Y_test):
    dummy_stratified = DummyClassifier()
    dummy_stratified.fit(X_train, Y_train)
    dummy_accuracy = dummy_stratified.score(X_test, Y_test) * 100
    print(f"Dummy stratified accuracy: {dummy_accuracy}")


def decision_tree(X_train, X_test, Y_train, Y_test):
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(X_train, Y_train)
    accuracy = model.score(Y_test, X_test) * 100
    print(f"Decision tree accuracy: {accuracy}")
    return model


if __name__ == '__main__':
    SEED = 20
    np.random.seed(SEED)
    file_path = "/home/dadaia/workspace/learning/3_AI/0_datasets/classification/dadaia_carros.csv"
    df_raw_data = pd.read_csv(file_path)
    df_handled_data = wrangle_data(df_raw_data)
    X_train, X_test, Y_train, Y_test = select_features_and_train(df_handled_data)
    baseline_dummy(X_train, X_test, Y_train, Y_test)
    model = decision_tree(X_train, X_test, Y_train, Y_test)