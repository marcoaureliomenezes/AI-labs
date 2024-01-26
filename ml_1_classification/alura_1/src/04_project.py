import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot as plt


def read_csv(file_path, delimiter=';'):
    with open(file_path, 'r') as f:
        yield from [row for row in csv.reader(f, delimiter=delimiter)]

# passing header columns reading csv in pandas


if __name__ == '__main__':

    file_path = '../files/freelances.csv'
    header = ['nao_finalizado', 'horas_esperadas', 'preco']
    features = ['horas_esperadas', 'preco']
    target = 'nao_finalizado'
    SEEDED = 20
    np.random.seed(SEEDED)
    
    df_data = pd.read_csv(file_path, delimiter=',', header=0, names=header)
    df_data['finalizado'] = df_data.nao_finalizado.map({0: 1, 1: 0})
    df_x = df_data[features]
    df_y = df_data[target]


    raw_train_x, raw_test_x, train_y, test_y = train_test_split(df_x, df_y, random_state=SEEDED, test_size=0.25, stratify=df_y)

    scaler = StandardScaler()
    scaler.fit(raw_train_x)
    train_x = scaler.transform(raw_train_x)
    test_x = scaler.transform(raw_test_x)


    print(len(train_x))
    print(len(test_x))

    modelo = SVC(gamma='auto')
    modelo.fit(train_x, train_y)
    predict = modelo.predict(test_x)
    accuracy = accuracy_score(test_y, predict)

    print("Accuracy of the model: ", round(accuracy * 100, 2), "%")

    baseline = np.ones(540)
    accuracy_bad = accuracy_score(test_y, baseline)
    print("Accuracy of the baseline: ", round(accuracy_bad * 100, 2), "%")
    # sns.relplot(x='horas_esperadas', y='preco', hue="finalizado", col="finalizado", data=df_data)

    data_x = train_x[:, 0]
    data_y = train_x[:, 1]

    x_min = data_x.min()
    x_max = data_x.max()
    y_min = data_y.min()
    y_max = data_y.max()

    pixels = 100
    axis_x = np.arange(x_min, x_max, (x_max - x_min) / pixels)
    axis_y = np.arange(y_min, y_max, (y_max - y_min) / pixels)
    xx, yy = np.meshgrid(axis_x, axis_y)
    pontos = np.c_[xx.ravel(), yy.ravel()]

    Z = modelo.predict(pontos)
    Z = Z.reshape(xx.shape)
    print(Z)

    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(data_x, data_y, c=train_y, s=1.5)
    plt.show()

