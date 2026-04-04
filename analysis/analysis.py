# %%

# 1. Imports e Configurações

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
# %%

# 2. Carregamento dos dados

df = pd.read_csv('../data/tripadvisor_hotel_reviews.csv')

# %%

df.head()

# %%

df.shape

# %%

df.info()

# %%

# 3. Análise Exploratória (EDA)

# Checar a distribuição das avaliações

df['Rating'].value_counts()

# Dataset desbalanceado, com a maioria das avaliações sendo 5 estrelas.

# %%

# Criar coluna sentimento no dataset

df['Sentiment'] = df['Rating'].map(lambda x: 'Positivo' if x >= 4 else 'Negativo' if x <= 2 else 'Neutro')
df['Sentiment'].value_counts()
# %%

# Porcentagem e visualização da distribuição dos sentimentos

porcentagem_sentimentos = df['Sentiment'].value_counts(normalize=True) * 100
porcentagem_sentimentos.head()
# %%

sns.countplot(x='Sentiment', data=df, order=['Positivo', 'Neutro', 'Negativo'])

ax = plt.gca()
ax.bar_label(ax.containers[0])

plt.title('Distribuição dos Sentimentos nas Avaliações')
plt.xlabel('Sentimento')
plt.ylabel('Contagem')

plt.tight_layout()
plt.savefig('../img/distribuicao_sentimentos.png')

plt.show()

# %%
