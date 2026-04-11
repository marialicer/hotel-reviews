# Análise de Sentimentos em Avaliações de Hotéis - HospedaAI

## Contexto de Negócio

A HospedaAI é uma consultoria de dados especializada no setor hoteleiro. Seu modelo de negócio consiste em coletar avaliações de plataformas como TripAdvisor, Booking e Google e entregar relatórios e modelos preditivos que ajudam gestores a tomar decisões mais inteligentes sobre a experiência dos hóspedes.

Um dos clientes da HospedaAI (uma rede com 8 hotéis no Sul do Brasil) identificou uma queda consistente no NPS nos últimos trimestres, mas não conseguia apontar a causa. Com mais de 500 novas avaliações por mês distribuídas entre as unidades, a leitura manual se tornou inviável.

## Problema de Negócio

A equipe de gestão da rede não sabe o que exatamente está afastando hóspedes (se é limpeza, atendimento, localização, café da manhã ou outro fator). Sem essa informação, qualquer ação de melhoria é baseada em intuição, não em dados.

A HospedaAI foi contratada para automatizar a leitura dessas avaliações e entregar um modelo capaz de classificar o sentimento de cada review, identificando padrões que expliquem a queda no NPS.

## Objetivos

- Realizar uma Análise Exploratória de Dados (EDA) para entender a distribuição das avaliações e padrões de texto
- Construir um pipeline de pré-processamento de texto reutilizável
- Comparar abordagens de análise de sentimentos: léxico-baseada vs. supervisionada
- Avaliar modelos clássicos de ML e transformers para classificação de sentimento
- Traduzir os resultados em insights de negócio para a equipe de gestão da rede hoteleira

---

## Dataset

- **Fonte:** [Trip Advisor Hotel Reviews - Kaggle](https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews)
- **Volume:** ~20.000 avaliações em inglês
- **Colunas:** `Review` (texto da avaliação) e `Rating` (nota de 1 a 5 estrelas)
- **Tarefa:** Classificação binária de sentimento - Positivo (ratings 3, 4 e 5) e Negativo (ratings 1 e 2)

---

## Metodologia

Este projeto segue a metodologia **CRISP-DM** (Cross-Industry Standard Process for Data Mining).

### 1. Análise Exploratória de Dados

- Distribuição dos ratings e sentimentos
- Comprimento médio das avaliações por sentimento. Reviews negativas têm em média **120 palavras** contra **101 das positivas**
- Nuvens de palavras por sentimento com stopwords customizadas para o domínio hoteleiro

**Principais achados:**
- 75% das avaliações são positivas (dataset desbalanceado)
- Hóspedes insatisfeitos escrevem reviews mais longas e detalhadas
- Reviews positivas destacam infraestrutura (pool, beach) e hospitalidade (friendly, breakfast)
- Reviews negativas concentram falhas de processo (told, asked), higiene (dirty) e barulho (noisy)

### 2. Pré-processamento de Texto

Pipeline aplicado sobre a coluna `Review`:

- Lowercase
- Remoção de pontuação e caracteres especiais com `re`
- Remoção de stopwords e mantendo termos de negação (`not`, `no`, `nor`)
- Tokenização com NLTK
- Lemmatização com `WordNetLemmatizer`


### 3. Vetorização

- **TF-IDF** com `max_features=5000` e `ngram_range=(1,2)`
- Fit exclusivamente no conjunto de treino para evitar **data leakage**

### 4. Modelos Avaliados

| Modelo | Tipo | Recall Negativo |
|--------|------|----------------|
| VADER | Léxico (baseline) | 0.42 |
| Naive Bayes | Supervisionado | 0.59 |
| Logistic Regression (sem balanceamento) | Supervisionado | 0.69 |
| SVM | Supervisionado | 0.76 |
| **Logistic Regression (balanceada)** | **Supervisionado** | **0.92** |
| BERT (DistilBERT pré-treinado) | Transformer | 0.93 |

---

## Resultados

O modelo selecionado foi a **Regressão Logística com `class_weight='balanced'`**.

| Métrica | Negativo | Positivo |
|---------|----------|----------|
| Precision | 0.69 | 0.99 |
| Recall | 0.92 | 0.93 |
| F1-Score | 0.79 | 0.95 |
| **Accuracy** | | **0.93** |

---

## Decisões de Negócio

**Por que não o BERT?**
O DistilBERT pré-treinado atingiu recall 0.93 para negativos, empatando com a Regressão Logística. Porém sua precisão foi de apenas 0.44 (quase metade dos alertas seriam falsos alarmes para a equipe da HospedaAI). Além disso, o modelo foi treinado em tweets do Twitter, um domínio diferente de reviews hoteleiras (*domain shift*). Para operações que exigem eficiência e baixo volume de alertas falsos, a Regressão Logística oferece melhor equilíbrio.

**Por que recall e não acurácia?**
A versão sem balanceamento atingiu 94% de acurácia (maior que o modelo escolhido). Porém seu recall para negativos caiu para 0.69, significando que 31% das reclamações reais passariam despercebidas. Para o objetivo da HospedaAI, **não detectar uma reclamação é pior do que investigar um falso alarme**.

---

## Impacto para o Negócio

Com o modelo em produção, a HospedaAI será capaz de entregar à rede hoteleira:

- Classificação automática de sentimento para cada nova avaliação recebida
- Identificação dos temas que mais impactam negativamente a experiência do hóspede
- Monitoramento do NPS por unidade ao longo do tempo
- Base para ações corretivas priorizadas por impacto

---

## Ferramentas Utilizadas

- **Manipulação de dados:** Pandas, NumPy
- **Visualização:** Matplotlib, Seaborn, WordCloud
- **NLP:** NLTK, VADER, spaCy
- **Machine Learning:** Scikit-learn (Logistic Regression, Naive Bayes, SVM), TF-IDF
- **Transformers:** HuggingFace Transformers (DistilBERT)
- **Métricas:** Accuracy, Precision, Recall, F1-Score, Classification Report

---

## Autora

Maria Alice Rocha<br>
Jornalista e pós graduada em Analytics e Business Intelligence<br>
Foco em análise de dados, storytelling, ciência de dados e insights acionáveis