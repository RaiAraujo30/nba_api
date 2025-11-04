# Atividade NBA - Regressão Linear e Logística

Sistema para análise de regressão linear e logística usando dados da NBA API.

## 📋 Descrição

Este projeto implementa modelos de regressão linear e logística para prever estatísticas de jogadores e times da NBA usando dados da temporada 2024-25 (ou outras temporadas disponíveis).

## 🚀 Instalação

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Executar aplicação

```bash
streamlit run app.py
```

A aplicação será aberta automaticamente no navegador (geralmente em `http://localhost:8501`).

## 📁 Estrutura do Projeto

```
Atividade_nba/
├── __init__.py              # Inicialização do módulo
├── app.py                   # Interface Streamlit principal
├── data_collector.py        # Coleta de dados da NBA API
├── feature_engineering.py   # Criação de features
├── regression_models.py    # Modelos de regressão
├── visualizations.py        # Visualizações
├── requirements.txt         # Dependências
└── README.md               # Este arquivo
```

## 🎯 Funcionalidades

### 1. Coleta de Dados
- Busca de jogadores e times por nome
- Download de dados de jogos da temporada
- Suporte para múltiplas temporadas

### 2. Feature Engineering
- Criação automática de features baseadas em dados históricos
- Médias, desvios padrão, máximos, mínimos
- Estatísticas dos últimos N jogos
- Features de contexto (jogo anterior, etc.)

### 3. Regressão Linear
- Seleção customizada de variáveis dependentes e independentes
- Treinamento e avaliação de modelos
- Visualizações:
  - Diagrama de Dispersão com Linha de Regressão
  - Previsão vs. Realidade
  - Tendência com Intervalo de Confiança
  - Análise de Resíduos

### 4. Regressão Logística
- Classificação binária de estatísticas
- Matriz de Confusão
- Relatório de Classificação
- Threshold configurável

## 📊 Variáveis Disponíveis

### Variáveis Dependentes (Y):
- **Pontos** (PTS)
- **Rebotes** (REB)
- **Assistências** (AST)

### Variáveis Independentes (X):
- Médias da temporada: `avg_pts`, `avg_reb`, `avg_ast`, etc.
- Médias dos últimos 5 jogos: `pts_last_5`, `reb_last_5`, `ast_last_5`
- Desvios padrão: `std_pts`, `std_reb`, `std_ast`
- Estatísticas do jogo anterior: `prev_pts`, `prev_reb`, `prev_ast`
- Médias de arremessos: `avg_fgm`, `avg_fga`, `avg_fg_pct`
- Médias de arremessos de 3: `avg_fg3m`, `avg_fg3a`, `avg_fg3_pct`
- E muitas outras...

## 🎮 Como Usar

1. **Selecionar tipo de análise**: Jogador ou Time
2. **Buscar entidade**: Digite o nome e clique em "Buscar"
3. **Carregar dados**: Clique em "Carregar Dados"
4. **Criar features**: Clique em "Criar Features"
5. **Treinar modelo**:
   - Selecione variável dependente (Y)
   - Selecione variáveis independentes (X)
   - Clique em "Treinar Modelo"
6. **Visualizar resultados**: Gráficos e métricas serão exibidos automaticamente

## 📈 Hipóteses Suportadas

### Para Jogadores:
- Um determinado Jogador fará Y pontos?
- Um determinado Jogador fará Y rebotes?
- Um determinado Jogador fará Y assistências?

### Para Times:
- O time fará "X Pontos" no jogo?
- O time fará "X Rebotes" no jogo?
- O time fará "X Assistências" no jogo?

## ⚙️ Configurações

- **Temporada**: Selecionável (2024-25, 2023-24, etc.)
- **Tipo de análise**: Jogador ou Time
- **Variáveis**: Seleção livre de variáveis dependentes e independentes

## 📝 Notas

- A API da NBA pode ter rate limiting. Aguarde alguns segundos entre requisições.
- Certifique-se de que a temporada selecionada já começou e tem dados disponíveis.
- Quanto mais jogos disponíveis, melhor será a qualidade das features e do modelo.

## 🔧 Dependências

- `streamlit`: Interface web
- `pandas`: Manipulação de dados
- `numpy`: Operações numéricas
- `scikit-learn`: Modelos de machine learning
- `matplotlib`: Visualizações
- `seaborn`: Visualizações avançadas
- `nba-api`: API para dados da NBA

## 📚 Referências

- [NBA API Documentation](https://github.com/swar/nba_api)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)

## 👨‍💻 Autor

Desenvolvido para a atividade de Redes Neurais Artificiais - 2025.2

