# -*- coding: utf-8 -*-
import graphviz
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from sklearn.tree import export_graphviz

SEED = 300
np.random.seed(SEED)


def extract_data():
   uri = "https://gist.githubusercontent.com/guilhermesilveira/e99a526b2e7ccc6c3b70f53db43a87d2/raw/1605fc74aa778066bf2e6695e24d53cf65f2f447/machine-learning-carros-simulacao.csv"
   df = pd.read_csv(uri).drop(columns=["Unnamed: 0"], axis=1)
   return df


def generate_x_y(df):
    df_x = df[["preco", "idade_do_modelo","km_por_ano"]]
    df_y = df["vendido"]
    return df_x, df_y


def print_results(results):
    media = results['test_score'].mean()
    desvio_padrao = results['test_score'].std()
    print(f"Média {round(media * 100, 2)} - Intervalo [{round((media - 2 * desvio_padrao)*100, 2)}, {round((media + 2 * desvio_padrao) * 100, 2)}]")


def generate_modelo_col(df):
    df['modelo'] = df.idade_do_modelo + np.random.randint(-2, 3, size=df.shape[0])
    df.modelo = df.modelo + abs(df.modelo.min()) + 1
    return df


def generate_baselines(x, y):
    print("BASELINES")
    modelo_dummy_clasifier = DummyClassifier()
    results_dummy_classifier = cross_validate(modelo_dummy_clasifier, x, y, cv=10, return_train_score=False)
    print_results(results_dummy_classifier)

    modelo_decision_tree = DecisionTreeClassifier(max_depth=2)
    results_decision_tree = cross_validate(modelo_decision_tree, x, y, cv=10, return_train_score=False)
    print_results(results_decision_tree)


def investigate_decision_tree(df, x, y):
    model = DecisionTreeClassifier(max_depth=2)
    cv = GroupKFold(n_splits = 10)
    results = cross_validate(model, x, y, cv = cv, groups = df.modelo, return_train_score=False)
    print_results(results)


def investigate_svc(df, x, y):
    scaler = StandardScaler()
    modelo_svc = SVC()
    pipeline_svc = Pipeline([('transformacao', scaler), ('estimador',modelo_svc)])
    cv = GroupKFold(n_splits = 10)
    results_svc = cross_validate(pipeline_svc, x, y, cv = cv, groups = df.modelo, return_train_score=False)
    print_results(results_svc)


if __name__ == "__main__":
    df_data = extract_data()
    df_x, df_y = generate_x_y(df_data)
    df_data_azar = df_data.sort_values("vendido", ascending=True)
    x_azar, y_azar = generate_x_y(df_data_azar)

    print(df_data_azar.head())

    df_data_modelo = generate_modelo_col(df_data) 
    print(df_data_modelo.head())

    generate_baselines(x_azar, y_azar)
    investigate_decision_tree(df_data, df_x, df_y)
    investigate_svc(df_data, x_azar, y_azar)


    modelo_decision_tree = DecisionTreeClassifier(max_depth=2)
    modelo_decision_tree.fit(x_azar, y_azar)
    dot_data = export_graphviz(modelo_decision_tree, out_file=None, filled=True, rounded=True, class_names=['não', 'sim'], feature_names = x_azar.columns)
    graph = graphviz.Source(dot_data)
    graph.render('dtree_render', view=True)