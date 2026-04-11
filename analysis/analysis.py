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
nltk.download('vader_lexicon')

from nltk.corpus import stopwords
from wordcloud import WordCloud

import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('omw-1.4')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from transformers import pipeline

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

df['Sentiment'] = df['Rating'].map(lambda x: 'Positivo' if x >= 3 else 'Negativo')
df['Sentiment'].value_counts()
# %%

# Porcentagem e visualização da distribuição dos sentimentos

porcentagem_sentimentos = df['Sentiment'].value_counts(normalize=True) * 100
porcentagem_sentimentos.head()
# %%

sns.countplot(x='Sentiment', data=df, order=['Positivo', 'Negativo'])

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
# Reviews negativas têm em média 120 palavras contra 101 das positivas

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

# 5. Vetorizar e treinar o modelo

X = df['Review_Clean']
y = df['Sentiment']
# %%

X_train, X_test, y_train, y_test = train_test_split (
    X, y, test_size=0.2, random_state=42
)
# %%

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
# %%

# fit apenas no treino para evitar data leakage

X_train_tfidf = vectorizer.fit_transform(X_train)
# %%

X_test_tfidf = vectorizer.transform(X_test)
# %%

# treinar o modelo
# a amostra está muito desbalanceada, tratamos o desbalanceamento

model = LogisticRegression(class_weight='balanced')
model.fit(X_train_tfidf, y_train)
# %%
y_pred = model.predict(X_test_tfidf)
# %%

# avaliar o modelo de regressão logistica

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# %%

print(classification_report(y_test, y_pred))

# modelo atinge 93% de acurácia, com excelente desempenho na classe positiva e alto recall para negativos.
# importante para capturar avaliações insatisfeitas

# %%

# teste sem balanceamento

model_unbalanced = LogisticRegression()
model_unbalanced.fit(X_train_tfidf, y_train)
# %%
y_pred = model_unbalanced.predict(X_test_tfidf)
# %%

# avaliar o modelo de regressão logistica

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
# %%

print(classification_report(y_test, y_pred))

# sem balanceamento obtive maior accuracy (94%), mas com perda significativa de recall para a classe negativa
# com base no objetivo do negócio, maximizar o recall da classe negativa é mais importante do que a acurácia geral, já que perder avaliações negativas significaria 
# ignorar potenciais problemas operacionais. Dessa forma, optei por utilizar o modelo com class_weight='balanced'
# %%

# 6. Comparar abordagens léxico-baseada x supervisionada

sia = SentimentIntensityAnalyzer()

df['vader_score'] = df['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])

df['vader_sentiment'] = df['vader_score'].apply(
    lambda x: 'Positivo' if x >= 0 else 'Negativo'
)
# %%

print(classification_report(df['Sentiment'], df['vader_sentiment']))

# apesar de uma boa acurácia geral (89%), 
# o modelo apresenta baixo recall para a classe negativa, 
# falhando em identificar a maioria das avaliações insatisfeitas, 
# o que compromete seu uso em cenários de monitoramento de qualidade

# %%

# 7. Testar modelos clássicos de ML

# Naive Bayes

nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
# %%

y_pred = nb.predict(X_test_tfidf)

# %%

# avaliar o modelo naive bayes

print(classification_report(y_test, y_pred))

# %%

#SVM

svm = LinearSVC()
svm.fit(X_train_tfidf, y_train)
# %%

y_pred = svm.predict(X_test_tfidf)

# %%

# avaliar o modelo SVM

print(classification_report(y_test, y_pred))

# %%

# 8. Testar BERT pré-treinado

# criar pipeline
model = pipeline("sentiment-analysis")

# aplicar no dataset (retorna label e score)
def get_label(text):
    result = model(text, truncation=True)[0]
    # mapear para o padrão do dataset
    return "Negativo" if result['label'] == 'NEGATIVE' else "Positivo"

# criar coluna com previsão do BERT
df['bert_sentiment'] = df['Review'].apply(get_label)
# %%
df.head()
# %%
print(classification_report(df['Sentiment'], df['bert_sentiment']))
print(accuracy_score(df['Sentiment'], df['bert_sentiment']))

# O modelo baseado em transformer (DistilBERT) apresentou o maior recall para a classe negativa (93%), 
# demonstrando alta capacidade de identificar avaliações insatisfeitas. 
# No entanto, apresentou baixa precisão (44%), indicando elevado número de falsos positivos.

# Em cenários onde é crítico não perder clientes insatisfeitos, o uso de modelos baseados em transformers é recomendado
# Entretanto, para operações que exigem maior eficiência e menor volume de alertas falsos, 
# modelos como Logistic Regression podem oferecer melhor equilíbrio

# %%

# visualizando e comparando o valor do recall dos modelos

recall_neg = {
    'VADER': 0.42,
    'Logistic Balanced': 0.92,
    'Logistic Unbalanced': 0.69,  
    'Naive Bayes': 0.59,
    'SVM': 0.76,
    'BERT': 0.93
}
# %%

# nomes dos modelos
modelos = list(recall_neg.keys())

# valores de recall
valores = list(recall_neg.values())
# %%

plt.figure(figsize=(8,5))
plt.bar(modelos, valores)
plt.ylim(0, 1)
plt.ylabel('Recall Classe Negativa')
plt.title('Capacidade dos Modelos em Detectar Clientes Insatisfeitos (Recall Negativo)')
plt.xticks(rotation=45)

for i, v in enumerate(valores):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

plt.savefig('../img/comparacao_recall_modelos.png')

plt.show()
# %%

#Conclusão da comparação:
#Para o objetivo da HospedaAI, a Regressão Logística Balanceada continua sendo a melhor escolha,
# mesmo recall que o BERT, mas com precision muito superior. 
# Além de ter sido treinada nos próprios dados do problema.