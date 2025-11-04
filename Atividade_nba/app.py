"""
Interface Streamlit para Análise de Regressão NBA
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Adicionar diretório atual ao path para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Importar módulos do mesmo diretório
from data_collector import NBADataCollector
from feature_engineering import FeatureEngineer
from regression_models import LinearRegressionModel, LogisticRegressionModel
from visualizations import RegressionVisualizer

# Configuração da página
st.set_page_config(
    page_title="NBA Regression Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🏀 Análise de Regressão NBA")
st.markdown("---")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de tipo de entidade
entity_type = st.sidebar.radio(
    "Selecione o tipo de análise:",
    ["Jogador", "Time"]
)

# Seleção de temporada
season = st.sidebar.selectbox(
    "Temporada:",
    ["2024-25", "2023-24", "2022-23"],
    index=0
)

# Inicializar coletor de dados
collector = NBADataCollector(season=season)
feature_engineer = FeatureEngineer()

# Busca de entidade
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Buscar " + entity_type)

if entity_type == "Jogador":
    player_name = st.sidebar.text_input("Nome do jogador:", placeholder="Ex: LeBron James")
    
    if st.sidebar.button("Buscar Jogador"):
        with st.spinner("Buscando jogador..."):
            player = collector.find_player(player_name)
            if player:
                st.session_state['selected_entity'] = player
                st.session_state['entity_type'] = 'player'
                st.sidebar.success(f"Jogador encontrado: {player['full_name']}")
            else:
                st.sidebar.error("Jogador não encontrado!")
else:
    team_name = st.sidebar.text_input("Nome do time:", placeholder="Ex: Los Angeles Lakers, Lakers, LAL")
    st.sidebar.caption("💡 Você pode buscar por nome completo, cidade, apelido ou abreviação")
    
    if st.sidebar.button("Buscar Time"):
        with st.spinner("Buscando time..."):
            if not team_name or team_name.strip() == "":
                st.sidebar.warning("⚠️ Digite o nome de um time")
            else:
                team = collector.find_team(team_name.strip())
                if team:
                    st.session_state['selected_entity'] = team
                    st.session_state['entity_type'] = 'team'
                    st.sidebar.success(f"✅ Time encontrado: {team['full_name']}")
                    st.sidebar.info(f"ID: {team['id']} | Abreviação: {team['abbreviation']}")
                else:
                    st.sidebar.error(f"❌ Time '{team_name}' não encontrado!")
                    st.sidebar.info("💡 Tente buscar por:\n"
                                   "- Nome completo (ex: 'Los Angeles Lakers')\n"
                                   "- Cidade (ex: 'Los Angeles')\n"
                                   "- Apelido (ex: 'Lakers')\n"
                                   "- Abreviação (ex: 'LAL')")

# Se uma entidade foi selecionada
if 'selected_entity' in st.session_state:
    entity = st.session_state['selected_entity']
    entity_type = st.session_state['entity_type']
    
    st.header(f"📊 Análise: {entity['full_name']}")
    
    # Carregar dados
    if st.button("🔄 Carregar Dados", type="primary"):
        with st.spinner("Carregando dados da NBA..."):
            try:
                if entity_type == 'player':
                    df = collector.get_player_game_log(entity['id'])
                else:
                    df = collector.get_team_game_log(entity['id'])
                
                # Verificar se os dados foram carregados corretamente
                if df is not None:
                    if len(df) > 0:
                        st.session_state['game_log'] = df
                        st.session_state['features_df'] = None
                        st.success(f"✅ Dados carregados com sucesso! {len(df)} jogos encontrados.")
                        st.rerun()  # Recarregar a página para mostrar os dados
                    else:
                        # Verificar se é problema de temporada ou realmente não há dados
                        st.warning(f"⚠️ Dados carregados, mas nenhum jogo encontrado para {entity['full_name']} na temporada {season}.")
                        st.info(f"💡 **Possíveis causas:**\n"
                               f"- A temporada {season} pode ainda não ter começado ou não ter dados disponíveis na API\n"
                               f"- Tente usar uma temporada anterior (ex: 2023-24)\n"
                               f"- Verifique se o {'jogador' if entity_type == 'player' else 'time'} jogou na temporada {season}\n"
                               f"- A API da NBA pode ter atraso na disponibilização dos dados")
                        st.session_state['game_log'] = df  # Salvar mesmo se vazio para debug
                        
                        # Mostrar preview do DataFrame vazio para debug
                        with st.expander("🔍 Debug: Ver DataFrame vazio"):
                            st.write(f"DataFrame shape: {df.shape}")
                            st.write(f"Colunas: {df.columns.tolist() if hasattr(df, 'columns') else 'N/A'}")
                else:
                    st.error(f"❌ Não foi possível carregar os dados. Verifique se a temporada {season} está disponível para {entity['full_name']}.")
                    st.info(f"💡 **Dicas:**\n"
                           f"- Verifique se a temporada {season} existe e tem dados disponíveis\n"
                           f"- Tente usar uma temporada anterior (ex: 2023-24)\n"
                           f"- Verifique sua conexão com a internet\n"
                           f"- Verifique os logs no console para mais detalhes")
            except Exception as e:
                st.error(f"❌ Erro ao carregar dados: {str(e)}")
                with st.expander("🔍 Ver detalhes do erro"):
                    st.exception(e)
    
    # Se os dados foram carregados
    if 'game_log' in st.session_state:
        df = st.session_state['game_log']
        
        # Mostrar informações sobre os dados carregados
        if df is not None and len(df) > 0:
            st.info(f"✅ {len(df)} jogos carregados para {entity['full_name']} na temporada {season}")
        elif df is not None:
            st.warning(f"⚠️ Dados carregados, mas nenhum jogo encontrado para {entity['full_name']} na temporada {season}")
        
        # Criar features
        if st.button("🔧 Criar Features"):
            with st.spinner("Criando features..."):
                if entity_type == 'player':
                    features_df = feature_engineer.create_player_features(df)
                else:
                    features_df = feature_engineer.create_team_features(df)
                
                if features_df is not None and not features_df.empty:
                    st.session_state['features_df'] = features_df
                    st.success(f"Features criadas! {len(features_df)} amostras disponíveis.")
                else:
                    st.error("Não foi possível criar features. Dados insuficientes.")
        
        # Mostrar dados mesmo sem features criadas
        if df is not None and len(df) > 0:
            st.markdown("---")
            
            # Tabs para diferentes análises
            tab1, tab2, tab3 = st.tabs(["📈 Regressão Linear", "📊 Regressão Logística", "📋 Dados"])
            
            # Verificar se features foram criadas
            if 'features_df' in st.session_state and st.session_state['features_df'] is not None:
                features_df = st.session_state['features_df']
            else:
                features_df = None
            
            with tab1:
                st.subheader("Regressão Linear")
                
                if features_df is not None and len(features_df) > 0:
                    # Seleção de variável dependente
                    target_var = st.selectbox(
                        "Variável Dependente (Y):",
                        ["target_pts", "target_reb", "target_ast"],
                        format_func=lambda x: {
                            "target_pts": "Pontos",
                            "target_reb": "Rebotes",
                            "target_ast": "Assistências"
                        }[x]
                    )
                    
                    # Seleção de variáveis independentes
                    available_features = feature_engineer.get_available_features(entity_type)
                    
                    selected_features = st.multiselect(
                        "Variáveis Independentes (X):",
                        available_features,
                        default=available_features[:5] if len(available_features) >= 5 else available_features
                    )
                    
                    if st.button("🚀 Treinar Modelo de Regressão Linear", type="primary"):
                        if len(selected_features) == 0:
                            st.warning("Selecione pelo menos uma variável independente!")
                        else:
                            with st.spinner("Treinando modelo..."):
                                # Preparar dados
                                X = features_df[selected_features].copy()
                                y = features_df[target_var].copy()
                                
                                # Remover valores nulos
                                mask = ~(X.isnull().any(axis=1) | y.isnull())
                                X = X[mask]
                                y = y[mask]
                                
                                if len(X) < 2:
                                    st.error("Dados insuficientes após limpeza!")
                                else:
                                    # Split treino/teste
                                    from sklearn.model_selection import train_test_split
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.2, random_state=42
                                    )
                                    
                                    # Treinar modelo
                                    model = LinearRegressionModel()
                                    model.fit(X_train, y_train, scale_features=True)
                                    
                                    # Avaliar
                                    train_results = model.evaluate(X_train, y_train)
                                    test_results = model.evaluate(X_test, y_test)
                                    
                                    # Armazenar resultados
                                    st.session_state['linear_model'] = model
                                    st.session_state['linear_X_train'] = X_train
                                    st.session_state['linear_y_train'] = y_train
                                    st.session_state['linear_X_test'] = X_test
                                    st.session_state['linear_y_test'] = y_test
                                    
                                    # Mostrar métricas
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("R² Score (Teste)", f"{test_results['r2']:.4f}")
                                    
                                    with col2:
                                        st.metric("RMSE (Teste)", f"{test_results['rmse']:.2f}")
                                    
                                    with col3:
                                        st.metric("MAE (Teste)", f"{test_results['mae']:.2f}")
                                    
                                    with col4:
                                        st.metric("MSE (Teste)", f"{test_results['mse']:.2f}")
                                    
                                    # Mostrar coeficientes
                                    st.subheader("Coeficientes do Modelo")
                                    coef = model.get_coefficients()
                                    
                                    coef_df = pd.DataFrame({
                                        'Variável': list(coef.keys()),
                                        'Coeficiente': list(coef.values())
                                    })
                                    st.dataframe(coef_df, width='stretch')
                                    
                                    # Visualizações
                                    st.subheader("Visualizações")
                                    
                                    visualizer = RegressionVisualizer()
                                    
                                    # Gráfico 1: Dispersão com linha de regressão
                                    fig1 = visualizer.plot_scatter_with_regression(
                                        X_test.iloc[:, 0] if len(selected_features) > 0 else X_test,
                                        y_test,
                                        test_results['predictions'],
                                        title="Diagrama de Dispersão com Linha de Regressão"
                                    )
                                    st.pyplot(fig1)
                                    
                                    # Gráfico 2: Previsão vs Realidade
                                    fig2 = visualizer.plot_prediction_vs_reality(
                                        y_test,
                                        test_results['predictions'],
                                        title="Previsão vs. Realidade"
                                    )
                                    st.pyplot(fig2)
                                    
                                    # Gráfico 3: Tendência com intervalo de confiança
                                    fig3 = visualizer.plot_trend_with_confidence(
                                        X_test.iloc[:, 0] if len(selected_features) > 0 else X_test,
                                        y_test,
                                        test_results['predictions'],
                                        model,
                                        title="Tendência com Intervalo de Confiança"
                                    )
                                    st.pyplot(fig3)
                                    
                                    # Gráfico 4: Resíduos
                                    fig4 = visualizer.plot_residuals(
                                        y_test,
                                        test_results['predictions'],
                                        title="Análise de Resíduos"
                                    )
                                    st.pyplot(fig4)
                else:
                    st.info("ℹ️ Crie as features primeiro clicando em 'Criar Features' para poder treinar modelos de regressão.")
            
            with tab2:
                st.subheader("Regressão Logística")
                
                if features_df is not None and len(features_df) > 0:
                    st.info("Para regressão logística, precisamos converter a variável dependente em binária.")
                    
                    # Seleção de variável dependente
                    target_var_log = st.selectbox(
                        "Variável Dependente (Y):",
                        ["target_pts", "target_reb", "target_ast"],
                        format_func=lambda x: {
                            "target_pts": "Pontos",
                            "target_reb": "Rebotes",
                            "target_ast": "Assistências"
                        }[x],
                        key="log_target"
                    )
                    
                    # Seleção de variáveis independentes
                    available_features_log = feature_engineer.get_available_features(entity_type)
                    
                    selected_features_log = st.multiselect(
                        "Variáveis Independentes (X):",
                        available_features_log,
                        default=available_features_log[:5] if len(available_features_log) >= 5 else available_features_log,
                        key="log_features"
                    )
                    
                    # Threshold para classificação binária
                    # Calcular max_value dinamicamente baseado nos dados
                    target_max = float(features_df[target_var_log].max())
                    target_min = float(features_df[target_var_log].min())
                    target_median = float(features_df[target_var_log].median())
                    
                    # Usar um valor maior que o máximo para permitir flexibilidade
                    max_threshold = max(100.0, target_max * 1.1) if target_max > 0 else 100.0
                    
                    threshold = st.number_input(
                        "Threshold para classificação binária:",
                        min_value=target_min,
                        max_value=max_threshold,
                        value=target_median,
                        step=1.0,
                        help=f"Valores disponíveis: min={target_min:.1f}, max={target_max:.1f}, mediana={target_median:.1f}"
                    )
                    
                    if st.button("🚀 Treinar Modelo de Regressão Logística", type="primary"):
                        if len(selected_features_log) == 0:
                            st.warning("Selecione pelo menos uma variável independente!")
                        else:
                            with st.spinner("Treinando modelo..."):
                                # Preparar dados
                                X = features_df[selected_features_log].copy()
                                y = features_df[target_var_log].copy()
                                
                                # Converter para binário
                                y_binary = (y > threshold).astype(int)
                                
                                # Remover valores nulos
                                mask = ~(X.isnull().any(axis=1) | y_binary.isnull())
                                X = X[mask]
                                y_binary = y_binary[mask]
                                
                                if len(X) < 2:
                                    st.error("Dados insuficientes após limpeza!")
                                else:
                                    # Split treino/teste
                                    from sklearn.model_selection import train_test_split
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
                                    )
                                    
                                    # Treinar modelo
                                    model = LogisticRegressionModel()
                                    model.fit(X_train, y_train, scale_features=True)
                                    
                                    # Avaliar
                                    train_results = model.evaluate(X_train, y_train)
                                    test_results = model.evaluate(X_test, y_test)
                                    
                                    # Armazenar resultados
                                    st.session_state['logistic_model'] = model
                                    st.session_state['logistic_X_test'] = X_test
                                    st.session_state['logistic_y_test'] = y_test
                                    
                                    # Mostrar métricas
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.metric("Accuracy (Teste)", f"{test_results['accuracy']:.4f}")
                                    
                                    with col2:
                                        st.metric("Threshold", f"{threshold:.1f}")
                                    
                                    # Mostrar coeficientes
                                    st.subheader("Coeficientes do Modelo")
                                    coef = model.get_coefficients()
                                    
                                    coef_df = pd.DataFrame({
                                        'Variável': list(coef.keys()),
                                        'Coeficiente': list(coef.values())
                                    })
                                    st.dataframe(coef_df, width='stretch')
                                    
                                    # Matriz de confusão
                                    st.subheader("Matriz de Confusão")
                                    visualizer = RegressionVisualizer()
                                    fig = visualizer.plot_confusion_matrix(
                                        y_test,
                                        test_results['predictions'],
                                        title="Matriz de Confusão"
                                    )
                                    st.pyplot(fig)
                                    
                                    # Classification report
                                    st.subheader("Relatório de Classificação")
                                    report_df = pd.DataFrame(test_results['classification_report']).transpose()
                                    st.dataframe(report_df, width='stretch')
                else:
                    st.info("ℹ️ Crie as features primeiro clicando em 'Criar Features' para poder treinar modelos de regressão logística.")
            
            with tab3:
                st.subheader("Dados")
                
                # Mostrar game log
                if df is not None and len(df) > 0:
                    st.write("**Game Log:**")
                    st.dataframe(df, width='stretch')
                    st.info(f"📊 Total de jogos: {len(df)}")
                else:
                    st.warning("⚠️ Nenhum dado de jogo disponível.")
                
                # Mostrar features
                if features_df is not None and len(features_df) > 0:
                    st.write("**Features Criadas:**")
                    st.dataframe(features_df, width='stretch')
                    
                    # Estatísticas descritivas
                    st.write("**Estatísticas Descritivas:**")
                    st.dataframe(features_df.describe(), width='stretch')
                else:
                    st.info("ℹ️ Features ainda não foram criadas. Clique em 'Criar Features' para gerar.")

else:
    st.info("👈 Use a barra lateral para buscar um jogador ou time e começar a análise.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Atividade de Redes Neurais Artificiais - Regressão Linear e Logística"
    "</div>",
    unsafe_allow_html=True
)

