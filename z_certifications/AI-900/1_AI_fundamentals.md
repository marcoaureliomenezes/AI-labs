# AI Fundamentals

## 1 - O que é Machine Learning

Fases:

- Treinamento;
- Inferência;
- Avaliação;

### 1.1 - Tipos de Machine Learning


### 1.1.1 - Apredizado Supervisionado

Nesse tipo de aprendizado os dados de treinamento contém features e labels. O objetivo é encontrar uma função que mapeie as features para os labels.

- `Classificação Binária`: o label é um valor booleano (0 ou 1);
- `Classificação Multiclasse`: o label é um valor discreto (0, 1, 2, 3, ...);
- `Regressão`: o label é um valor contínuo (0.1, 0.2, 0.3, ...);

Fases:

- Divisão dos dados randomicamente em treinamento e teste;
- Uso de algoritmos de treinamento e dados de treinamento para encontrar a função que mapeia as features para os labels;
- Uso de dados de teste para avaliar o modelo;
- Uso de algoritmos de avaliação para analisar o modelo a partir de métricas associadas de performance;



#### 1.1.1.1 - Regressão

- Treinados para prever um valor numérico baseado em um conjunto de features e labels (contínuos) de treinamento;


Métricas para avaliação de modelos de regressão:

- `Mean Absolute Error (MAE)`: média da diferença absoluta entre os valores previstos e os valores reais.
- `Mean Squared Error (MSE)`: média da diferença quadrática entre os valores previstos e os valores reais;
- `Root Mean Squared Error (RMSE)`: raiz quadrada da média da diferença quadrática entre os valores previstos e os valores reais;
- `Coefficient of Determination (R²)`: métrica que indica o quão bem os valores previstos se ajustam aos valores reais;


#### 1.1.1.2 - Classificação Binária

- Treinados para prever um valor booleano baseado em um conjunto de features e uma label binária de treinamento;

Métricas para avaliação de modelos de classificação binária:

- `Matriz de Confusão`: matriz que mostra a quantidade de acertos e erros do modelo;
- `Accuracy`: métrica que indica a proporção de acertos do modelo;
- `Precision`: métrica que indica a proporção de acertos positivos do modelo;
- `Recall`: métrica que indica a proporção de acertos positivos do modelo em relação ao total de positivos;
- `F1 Score`: média harmônica entre `Precision` e `Recall`;
- `Area Under the Curve (AUC)`: métrica que indica a probabilidade de o modelo classificar um exemplo positivo aleatório com uma probabilidade maior que um exemplo negativo aleatório;

#### 1.1.1.3 - Classificação Multi-classes

- Treinados para prever um valor discreto baseado em um conjunto de features e uma label discreta de treinamento;
- One vs All: treina um modelo para cada classe, onde a classe é considerada positiva e as demais negativas;
- One vs One: treina um modelo para cada par de classes, onde uma classe é considerada positiva e a outra negativa;


Métricas para avaliação de modelos de classificação multi-classes:

### 1.1.2 - Aprendizado Não Supervisionado

Nesse tipo de aprendizado os dados de treinamento contém apenas features. O objetivo é encontrar padrões nos dados e criar grupos de dados similares ou labels.

- Clustering
- Análise de Associação
- Redução de Dimensionalidade
