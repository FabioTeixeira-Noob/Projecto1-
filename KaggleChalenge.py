import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
# Supondo que o arquivo CSV esteja no mesmo diretório do script
data = pd.read_csv("Revisao Python\gym_churn_us.csv")

# Exibir as primeiras linhas do dataset
print(data.head())

# Informações gerais sobre o dataset
print(data.info())

# Estatísticas descritivas
print(data.describe())
# Distribuição do churn
sns.countplot(x='Churn', data=data)
plt.title('Distribuição do Churn')
plt.show()

# Distribuição de gênero
sns.countplot(x='gender', data=data)
plt.title('Distribuição de Gênero')
plt.show()

# Boxplot de idade por churn
sns.boxplot(x='Churn', y='Age', data=data)
plt.title('Idade vs Churn')
plt.show()

# Correlação entre variáveis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlação entre Variáveis')
plt.show()
# Converter variáveis categóricas em dummy/variáveis indicadoras
data = pd.get_dummies(data, columns=['gender', 'Near_Location', 'Partner', 'Promo_friends', 'Phone'], drop_first=True)

# Separar as variáveis independentes (X) e a variável dependente (y)
X = data.drop('Churn', axis=1)
y = data['Churn']

# Dividir o dataset em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Treinar o modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prever no conjunto de teste
y_pred = model.predict(X_test)
# Calcular as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f'Acurácia: {accuracy:.2f}')
print(f'Precisão: {precision:.2f}')
print(f'Recall: {recall:.2f}')

# Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Não Churn', 'Churn'], yticklabels=['Não Churn', 'Churn'])
plt.title('Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()
