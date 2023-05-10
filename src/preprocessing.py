# Importações de libraries importantes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle as pl
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler 
# Tema
sns.set_theme(style="ticks", palette="flare")


# Carregamento de dataframe
df = pd.read_csv("../dados/data.csv")


# Identificando valores null
print(pd.isna(df).sum())


# Identificando correlação entre features
sns.heatmap(df.corr(), cmap="mako")
print(df.corr())

# Função para plotar as distribuições das variáveis
def plota_dados_hist(coluna):

    sns.histplot(df[coluna])
    plt.show()

for coluna in df.columns:
    plota_dados_hist(coluna)


# One-hot encoding das variáveis categóricas
categoricas = ["AnnualIncomeClass", "ServicesOpted"]
df = pd.get_dummies(df, columns=categoricas)


colunas_one_encoding = ["FrequentFlyer", "AccountSyncedToSocialMedia", "BookedHotelOrNot"]
for coluna in colunas_one_encoding:
    
    df[coluna] = df[coluna].map({"Yes": 1, "No": 0})


# Normaliza variáveis numéricas
scaler = MinMaxScaler()
df["Age"] = scaler.fit_transform(df["Age"].to_numpy().reshape(-1, 1))

# Limpa dataset
df = df.dropna()

# Define X e Y
Y = df.iloc[:, 4:5].to_numpy()
X = df.drop("Target", axis=1).to_numpy()

"""
No dataset de treino, existe um evidente class imbalance que deve ser contornado 
para melhor performance do modelo. A técnica utilizada será o random oversampling, 
geração aleatória de datapoints gêmeos dos observados.
"""

# Random oversampling
sampler = RandomOverSampler(random_state=30)
X, Y = sampler.fit_resample(X, Y)


# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.15)


# Serialização de datasets
pl.dump(X_train, open("../dados/x_train.pkl", "wb"))
pl.dump(Y_train, open("../dados/y_train.pkl", "wb"))
pl.dump(X_test, open("../dados/x_test.pkl", "wb"))
pl.dump(Y_test, open("../dados/y_test.pkl", "wb"))
