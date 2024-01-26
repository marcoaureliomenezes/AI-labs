import numpy as np
import pandas as pd
import csv
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def read_csv(file_path, delimiter=';'):
    with open(file_path, 'r') as f:
        yield from [row for row in csv.reader(f, delimiter=delimiter)]

# passing header columns reading csv in pandas


if __name__ == '__main__':
    SEEDED = 50
    np.random.seed(SEEDED)

    
    file_path = '../files/tracking.csv'
    header = ['pagina_principal', 'como_funciona', 'contato', 'comprou']
    features = ['pagina_principal', 'como_funciona', 'contato']
    target = 'comprou'

    df_data = pd.read_csv(file_path, delimiter=',', header=0, names=header)
    df_x = df_data[features]
    df_y = df_data[target]

    train_x, test_x, train_y, test_y = train_test_split(df_x, df_y, random_state=SEEDED, test_size=0.25, stratify=df_y)


    modelo = LinearSVC()
    modelo.fit(train_x, train_y)
    predict = modelo.predict(test_x)
    accuracy = accuracy_score(test_y, predict)

    print("Accuracy: ", round(accuracy * 100, 2), "%")


