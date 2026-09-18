import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

from sklearn.datasets import load_wine
from sklearn.model_selection import (train_test_split, cross_val_score)
from sklearn.preprocessing import StandardScaler #normalização
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay)
from sklearn.tree import plot_tree

#%%
sns.set_theme(style="whitegrid")

#Carregando dataset

#%%
wine = load_wine()

#%%

#transformando o dataset em um daframe pra pandas
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)

df['target'] = wine.target

print(df.head())
#%%
#inspeção inicial
print("Dimensões:", df.shape)

print("Tipos:", df.dtypes)
print("Valores ausentes:", df.isna().sum())

#%%

print("quantidade de amostras: ", df.shape[0])
print("quantidade de colunas: ", df.shape[1])

print ("Primeira linhas: ", df.head())

#%%
#Descrever o Dataframe

print (df.describe())
#%%
for feature in wine.feature_names:
    print ("-", feature)
    
#%%
#Conhecendo o target

print ("Classes disponíveis: ", wine.target_names)

print ("Quantidade de amostras por classe")
print(df["target"].value_counts())

#%%
#gráfico

plt.figure(figsize=(8, 5))

sns.countplot(
    data = df,
    x = "target",
)

plt.title("Quantidade de amostras por classe")
plt.xlabel("Classe")
plt.ylabel("Quantidade de amostras")

plt.show()

#%%
#Gráfico intensidadeXteor
plt.figure(figsize=(8,5))

sns.scatterplot(
    data = df,
    x = "alcohol",
    y = "color_intensity",
    hue = "target",
)

plt.title("Relação teor alcólico e intensidade da cor")
plt.xlabel("teor alcóolico")
plt.ylabel("intensidade da cor")

plt.show()
#Se percebe padrão de agrupamento com parametro com potencial n arbitrário

# %%

#Matriz de Correlação
#a presença de um destaque na diagonal principal associa a correlação n causalidade
plt.figure(figsize=(11, 8))

sns.heatmap(
    df.drop(columns = "target").corr(),
    cmap = "coolwarm",
)

plt.title("Matriz de Correlação")

plt.show()

# %%
# Entre Features e Target pro modelo

X = df.drop(columns = "target") #Caracteristicas quimicas

y = df["target"] #classes do vinho
#Vai entrar no teste

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)

# %%

#Separando Treino e teste

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42, stratify=y)

# %%

# Escolhendo nosso primeiro Modelo : Decision Tree Classifier [varias arvores ponderando a decisão]
modelo_tree = DecisionTreeClassifier(
    max_depth=4,
    random_state=42
)

modelo_tree.fit(X_train, y_train)
previsoes_tree = modelo_tree.predict(X_test)

#%%previsão
#acurácia
accuracy_tree = accuracy_score(y_test, previsoes_tree)

print("Acurácia do modelo Decision Tree: ", accuracy_tree)

# %%
#Matriz de Confusão

ConfusionMatrixDisplay.from_predictions(y_test,
    previsoes_tree,
    display_labels = wine.target_names)

plt.title("Matriz de Confusão - Decision Tree Classifier")

plt.show()

# %%
#verificar a veracidade da previsão
#Descobrindo o numero de folha

profundidades = range(1, 11)

treino = []
teste = []

for profundidade in profundidades:
    modelo = DecisionTreeClassifier(max_depth = profundidade, random_state = 42)
    modelo.fit(X_train, y_train)
    treino.append(modelo.score(X_train, y_train))
    teste.append(modelo.score(X_test, y_test))

# %%
#Descobrindo a profundidade da folha

profundidades = range(1, 11)

treino = []
teste = []

for profundidade in profundidades:
    modelo = DecisionTreeClassifier(max_depth = profundidade, random_state = 42)
    modelo.fit(X_train, y_train)
    treino.append(modelo.score(X_train, y_train))
    teste.append(modelo.score(X_test, y_test))






# %%rf
modelo_RF = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

modelo_RF.fit(X_train, y_train)
previsoes_RF = modelo_RF.predict(X_test)

#%%previsão
#acurácia
accuracy_RF = accuracy_score(y_test, previsoes_RF)

print("Acurácia do modelo RandonF: ", accuracy_RF)

# %%
#Matriz de Confusão

ConfusionMatrixDisplay.from_predictions(y_test,
    previsoes_RF,
    display_labels = wine.target_names)

plt.title("Matriz de Confusão - RF")

plt.show()

# %%
#verificar a veracidade da previsão
#Descobrindo o numero de folha

profundidades = range(1, 11)

treino = []
teste = []

for profundidade in profundidades:
    modelo = RandomForestClassifier(max_depth = profundidade, random_state = 42)
    modelo.fit(X_train, y_train)
    treino.append(modelo.score(X_train, y_train))
    teste.append(modelo.score(X_test, y_test))

# %%
#Descobrindo a profundidade da folha

profundidades = range(1, 11)

treino = []
teste = []

for profundidade in profundidades:
    modelo = RandomForestClassifier(max_depth = profundidade, random_state = 42)
    modelo.fit(X_train, y_train)
    treino.append(modelo.score(X_train, y_train))
    teste.append(modelo.score(X_test, y_test))

#%%
# %%
#Comparando os modelos

resultados = pd.DataFrame({
    "Modelo": ["Decision Tree", "Random Forest"],
    "Acurácia": [accuracy_tree, accuracy_RF],
})

print(resultados)

# %%
#Plotando a arvore
print(classification_report(y_test, previsoes_RF, target_names=wine.target_names))

# %%
#gráficos de linhas de treino
plt.figure(figsize=(9, 5))

plt.plot(profundidades, treino, marker="o", label="Treino")
plt.plot(profundidades, teste, marker="o", label="Teste")

plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.title("Treino x teste por profundidade")
plt.xticks(range(1, 11))
plt.legend()
plt.show()


# %%
#Importância das features RF

# Criando a Series com as importâncias
importancias = pd.Series(
    modelo_RF.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("Importância das features:")
print(importancias)

# Plotando o gráfico
plt.figure(figsize=(10, 6))
importancias.head(10).sort_values().plot(kind="barh")
plt.title("10 features mais importantes (Random Forest)")
plt.xlabel("Importância")
plt.show()
