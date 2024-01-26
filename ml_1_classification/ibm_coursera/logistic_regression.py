from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import pandas
import seaborn as sns, pandas as pd, numpy as np


# Read Data
df_data = pd.read_csv("../../../0_datasets/classification/recognition_with_smartphone_data.csv", sep=',')
print(df_data.head())


# Explore Data

# Calculate the correlation values

le = LabelEncoder()
df_data['Activity'] = le.fit_transform(df_data.Activity)
df_data['Activity'].sample(5)


feature_cols = df_data.columns[:-1]
corr_values = df_data[feature_cols].corr()


# Prepare Data

# Split Data

strat_shuf_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)

train_idx, test_idx = next(strat_shuf_split.split(df_data[feature_cols], df_data.Activity))

# Create the dataframes
X_train = df_data.loc[train_idx, feature_cols]
y_train = df_data.loc[train_idx, 'Activity']

X_test  = df_data.loc[test_idx, feature_cols]
y_test  = df_data.loc[test_idx, 'Activity']

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

# Train Model

lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

lr_l1 = LogisticRegressionCV(Cs=10, cv=4, penalty='l1', solver='liblinear').fit(X_train, y_train)
lr_l2 = LogisticRegressionCV(Cs=10, cv=4, penalty='l2', solver='liblinear').fit(X_train, y_train)


coefficients = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    coeffs = mod.coef_
    coeff_label = pd.MultiIndex(levels=[[lab], [0,1,2,3,4,5]], codes=[[0,0,0,0,0,0], [0,1,2,3,4,5]])
    coefficients.append(pd.DataFrame(coeffs.T, columns=coeff_label))


coefficients = pd.concat(coefficients, axis=1)
coefficients.sample(10)


fig, axList = plt.subplots(nrows=3, ncols=2)
axList = axList.flatten()
fig.set_size_inches(10,10)

for ax in enumerate(axList):
    loc = ax[0]
    ax = ax[1]
    
    data = coefficients.xs(loc, level=1, axis=1)
    data.plot(marker='o', ls='', ms=2.0, ax=ax, legend=False)
    
    if ax is axList[0]:
        ax.legend(loc=4)
        
    ax.set(title='Coefficient Set '+str(loc))

plt.tight_layout()

### BEGIN SOLUTION
# Predict the class and the probability for each
y_pred = list()
y_prob = list()

coeff_labels = ['lr', 'l1', 'l2']
coeff_models = [lr, lr_l1, lr_l2]

for lab,mod in zip(coeff_labels, coeff_models):
    y_pred.append(pd.Series(mod.predict(X_test), name=lab))
    y_prob.append(pd.Series(mod.predict_proba(X_test).max(axis=1), name=lab))
    
y_pred = pd.concat(y_pred, axis=1)
y_prob = pd.concat(y_prob, axis=1)

y_pred.head()

y_prob.head()
### END SOLUTION




# Evaluate Model


# Predict Model



# LR = LogisticRegression(penalty='12', c=10.0)
# LR.fit(X_train, y_train)
# y_predict = LR.predict(X_test)