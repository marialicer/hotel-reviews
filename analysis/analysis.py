# %%

# 1. Imports e Configurações

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.corpus import stopwords
from wordcloud import WordCloud

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.feature_extraction.text import TfidfVectorizer

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

# O objetivo é descobrir se hóspedes insatisfeitos escrevem mais do que os satisfeitos 
# É uma hipótese intuitiva: quem reclama tende a se explicar mais

df['review_length'] = df['Review'].str.split().str.len()
df.head()
# %%

words_sentiment = df.groupby('Sentiment')['review_length'].mean()
print(words_sentiment)

# Resultado confirma hipótese:
# Hóspedes insatisfeitos escrevem mais
# Reviews negativas têm em média 120 palavras contra 99 das positivas

# %%

sns.boxplot(
    x='Sentiment',
    y= 'review_length',
    data=df
)

plt.ylim(0, 300)

plt.title('Distribuição dos Sentimentos nas Avaliações')
plt.xlabel('Sentimento')
plt.ylabel('Contagem')

plt.tight_layout()
plt.savefig('../img/distribuicao_palavras.png')

plt.show()

# A maioria das reviews positivas fica entre ~50 e ~115 palavras 
# Enquanto as negativas ficam entre ~60 e ~145
# Os pontos acima são outliers (algumas pessoas escreveram reviews gigantes independente do sentimento)

# %%

# Visualizar palavras aparecem nas reviews positivas e negativas 

df_positivos = df[df['Sentiment'] == 'Positivo']
texto_positivos = ' '.join(df_positivos['Review'])
# %%

# criar o objeto Wordcloud para palavras positivas

# Remover stopwords

remove_stopwords = set(stopwords.words('english'))

# remover palavras pouco úteis

remove_stopwords.update([
    'hotel', 'room', 'rooms', 'stay', 'stayed', 
    'one', 'would', 'could', 'also', 'like'
])

# %%
wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    stopwords=remove_stopwords,
    colormap='viridis',
    max_words=100
).generate(texto_positivos)
# %%

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('../img/nuvem_palavras_positivas.png')

plt.show()
# %%

df_negativos = df[df['Sentiment'] == 'Negativo']
texto_negativos = ' '.join(df_negativos['Review'])

# %%

# criar o objeto WordCloud para palavras negativas

wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    stopwords=remove_stopwords,
    colormap='viridis',
    max_words=100
).generate(texto_negativos)

# %%

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')

plt.savefig('../img/nuvem_palavras_negativas.png')

plt.show()

# Reviews Positivas: O hóspede exalta a infraestrutura (pool, beach), o sabor (breakfast) e a hospitalidade (friendly)
# Reviews Negativas: O hóspede detalha possíveis falhas de processo (told, asked), falta de higiene (dirty) e quebra de sossego (noisy)
# %%

# 4. Pré-processamento do texto

# restaurar stopwords para ML

stopwords_ml = set(stopwords.words('english'))

# manter negação 
stopwords_ml = stopwords_ml - {"not", "no", "nor"}

# %%

lemmatizer = WordNetLemmatizer()

def preprocess(text):

    # 1. lowercase
    text = text.lower()
    
    # 2. remove pontuação com re.sub()
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 3. tokenização
    words = word_tokenize(text)
    
    # 4. remove stopwords
    words = [word for word in words if word not in stopwords_ml]
    
    # 5. lemmatização de cada token
    words = [lemmatizer.lemmatize(word) for word in words]
    
    # 6. juntar tudo de volta
    texto_limpo = ' '.join(words)
    
    return texto_limpo
# %%

df['Review_Clean'] = df['Review'].apply(preprocess)
df[['Review', 'Review_Clean']].head()
# %%
