# Análise de Sentimentos em Avaliações de Hotéis — HospedaAI

## Contexto de Negócio

A HospedaAI é uma consultoria de dados especializada no setor hoteleiro. Seu modelo de negócio consiste em coletar avaliações de plataformas como TripAdvisor, Booking e Google e entregar relatórios e modelos preditivos que ajudam gestores a tomar decisões mais inteligentes sobre a experiência dos hóspedes.

Um dos clientes da HospedaAI (uma rede com 8 hotéis no Sul do Brasil) identificou uma queda consistente no NPS nos últimos trimestres, mas não conseguia apontar a causa. Com mais de 500 novas avaliações por mês distribuídas entre as unidades, a leitura manual se tornou inviável.

## Problema de Negócio

A equipe de gestão da rede não sabe o que exatamente está afastando hóspedes — se é limpeza, atendimento, localização, café da manhã ou outro fator. Sem essa informação, qualquer ação de melhoria é baseada em intuição, não em dados.

A HospedaAI foi contratada para automatizar a leitura dessas avaliações e entregar um modelo capaz de classificar o sentimento de cada review, identificando padrões que expliquem a queda no NPS.

## Objetivos

- Realizar uma Análise Exploratória de Dados (EDA) para entender a distribuição das avaliações e padrões de texto
- Construir um pipeline de pré-processamento de texto reutilizável
- Comparar abordagens de análise de sentimentos: léxico-baseada vs. supervisionada
- Avaliar modelos clássicos de ML e transformers para classificação de sentimento
- Traduzir os resultados em insights de negócio para a equipe de gestão da rede hoteleira