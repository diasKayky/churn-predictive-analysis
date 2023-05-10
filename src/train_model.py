# Importações de libraries importantes

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pickle as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Datasets
X_train = pl.load(open("../dados/x_train.pkl", "rb"))
X_test = pl.load(open("../dados/x_test.pkl", "rb"))
Y_train = pl.load(open("../dados/y_train.pkl", "rb"))
Y_test = pl.load(open("../dados/y_test.pkl", "rb"))
# Muda o formato dos arrays
Y_train = Y_train.reshape(-1)
Y_test = Y_test.reshape(-1)


"""
Treinamento
"""

# Modelo de ML
params = {
    "n_estimators": [10, 150, 300, 100, 700,  1000, 1500], "criterion": ["gini", "entropy"], 
    "max_features": ["sqrt", "log2"]}

clf = RandomForestClassifier()
grid = GridSearchCV(clf, params)
grid.fit(X_train, Y_train)


"""
Avaliação
"""

pred = grid.predict(X_test)

# Report de classificação
print(classification_report(pred, Y_test))

# Acurácia do Modelo
print(f"A acurácia do modelo é: {accuracy_score(pred, Y_test)}")

# Matriz de confusão
sns.heatmap(confusion_matrix(pred, Y_test), cmap="cividis")
print(confusion_matrix(pred, Y_test))


"""
Serializalção do Modelo
"""

# Serialização
modelo = grid.best_estimator_

pl.dump(modelo, open("../modelos/modelo.pkl", "wb"))