# Classificação

- Saida é uma categoria

# Necessidades para classificação

- Rotulos que são conhecidos
- Features que podem ser quantificadas
- Método para medir similaridade

## Modelos usados para Aprendizado supervisionado (Classificação e regressão)

- Logistic Regression
- K-Nearest Neighbors
- Support Vector Machines
- Neural Networks
- Decision Tree
- Random Forest
- Ensemble Models


### Aplicações para Regressão Logistica

- Customer spending: Caracterizar tendencias de compra de acordo com compras anteriores
- Customer engagement: Quem são os potenciais clientes nos próximos 6 meses.
- E-commerce: Quais transações são fraudulentas usando caracteristicas de click, location, IP address, etc.
- Finance/risk: Prever se emprestimo será pago.



### Introdução a Regressão Logística.


- Sigmoid Function

    y = 1 / (1 + e^-x)


### Regressão linear x Regressão Logística

### Usando regressão logistica para classificação


#### Erros de métrica em classificação

- Tipos de rros cometidos em problemas de classificação
- Abordagens para medir saídas de Classificação

Usando metricas de erros de classificação para escolher entre modelo.


**Exemplo:**

É requisitado que se contrua um classificador para prever se individuos terão leocemia

- Training Data: 1% de pacientes com leocemia e 99% saudáveis.
- Medição de acurácia: Número total de predições corretas
- Se um modelo simples sempre prevê "saudável", então a acurácia é de 99%.


Matriz de confusão

- True Positive (TP): Predição correta
- True Negative (TN): Predição correta
- False Positive (TP): Predição incorreta (Type I error)
- False Negative (TN): Predição incorreta (Type II error)


## Métricas
### Acuracia


$ accuracy = {(TP + TN) \over (TP + TN + FP + FN)} $


### Precisão

- Identifica somente instancias positivas

**Formula:**

$ precision = {(TP) \over (TP + FP)} $


### Recall or sensitivity: 

- Identifica todas as instancias positivas

**Formula:**

$ recall = {(TP) \over (TP + FN)} $

### Specificity: 

- Evitar alarmes Falsos.

**Formula:**

$ specificity = {(TN) \over (FP + TN)} $


### F1:

$ F1 = {2 * (Precision * Recall) \over (Precision + Recall)} $


### Receiver Operating Characteristic (ROC)

False Positive Rate (1 - specificity)

- Geralmente é bom para classes balanceadas

### Precision-recall Curve

- Geralmente é bom para classes balanceadas

Com mais de 2 classes

accuracy = TP1 + TP2 + TP3 / Tudo
As outras medidas devem ser medidas 1 classe vs todas


from sklearn.metrics import accuracy_score

accuracy
## Cross Validation



