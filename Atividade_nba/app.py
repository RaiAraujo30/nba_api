"""
Interface Streamlit para Análise de Regressão NBA
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        .sidebar-icon { margin-right: 8px; }
        .info-box {
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            font-size: 1rem;
            line-height: 1.6;
        }
        .info-box i {
            margin-right: 10px;
            font-size: 1.1rem;
        }
        .info-blue {
            background-color: #e3f2fd;
            color: #1565c0;
            border-left: 4px solid #2196F3;
        }
        .info-green {
            background-color: #e8f5e9;
            color: #2e7d32;
            border-left: 4px solid #4caf50;
        }
        .info-yellow {
            background-color: #fff8e1;
            color: #f57f17;
            border-left: 4px solid #ffc107;
        }
        .info-red {
            background-color: #ffebee;
            color: #c62828;
            border-left: 4px solid #f44336;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Botão primário Streamlit */
    button[kind="primary"] {
        background-color: #2196F3 !important;
        color: #fff !important;
        border: none !important;
        font-weight: bold;
        border-radius: 8px !important;
    }
    button[kind="primary"]:hover {
        background-color: #1565c0 !important;
        color: #fff !important;
    }
    /* Tab selector Streamlit */
    .stTabs [data-testid="stTab"] {
        color: #fff !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="true"] {
        color: #1565c0 !important;
        border-bottom: none !important;
    }
    .stTabs [aria-selected="false"] {
        border-bottom: none !important;
    }
    /* Remove barra vermelha extra (pseudo-elemento) */
    .stTabs [data-testid="stTab"]:after {
        background: transparent !important;
        border: none !important;
    }
    .st-av {
        background-color: #1565c0 !important;
    }
    </style>
""", unsafe_allow_html=True) 

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_collector import NBADataCollector
from feature_engineering import FeatureEngineer
from regression_models import LinearRegressionModel, LogisticRegressionModel
from visualizations import RegressionVisualizer

st.set_page_config(
    page_title="NBA Regression Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    '<h1><i class="fas fa-basketball-ball" style="color: #2196F3;"></i> Análise de Regressão NBA</h1>',
    unsafe_allow_html=True
)
st.markdown("---")

st.sidebar.markdown(
    '<h3><i class="fas fa-cog sidebar-icon" style="color: #2196F3;"></i>Configurações</h3>',
    unsafe_allow_html=True
)


st.markdown(
    """
    <style>
    [data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        border: 2px solid #2196F3 !important;
        background-color: #2196F3 !important;
    }
    /* Muda cor do texto */
    [data-testid="stRadio"] label {
        color: #fff !important;
        font-weight: bold;
    }
    </style>
""",
    unsafe_allow_html=True,
)

entity_type = st.sidebar.radio(
    "Selecione o tipo de análise:",
    ["Time (Golden State Warriors)", "Jogador (do Golden State Warriors)"],
)

season = st.sidebar.selectbox(
    "Temporada:",
    ["2024-25", "2023-24", "2022-23"],
    index=0
)

collector = NBADataCollector(season=season)
feature_engineer = FeatureEngineer()

st.sidebar.markdown("---")
st.sidebar.markdown(
    f'<h3><i class="fas fa-search sidebar-icon" style="color: #2196F3;"></i>Buscar</h3>',
    unsafe_allow_html=True,
)

if entity_type == "Time (Golden State Warriors)":
    if (
        "selected_entity" not in st.session_state
        or st.session_state.get("last_entity_type") != "team"
    ):
        team = collector.find_team("Golden State Warriors")
        if team:
            st.session_state["selected_entity"] = team
            st.session_state["entity_type"] = "team"
            st.session_state["last_entity_type"] = "team"
            st.sidebar.markdown(
                f'<div class="info-box info-green"><i class="fas fa-check-circle"></i>Time carregado: {team["full_name"]}</div>',
                unsafe_allow_html=True,
            )
    else:
        if (
            "selected_entity" in st.session_state
            and st.session_state["entity_type"] == "team"
        ):
            st.sidebar.markdown(
                f'<div class="info-box info-green"><i class="fas fa-check-circle"></i>Time carregado: {st.session_state["selected_entity"]["full_name"]}</div>',
                unsafe_allow_html=True,
            )
else:
    if 'gsw_roster' not in st.session_state:
        with st.spinner("Carregando roster do Golden State Warriors..."):
            st.session_state['gsw_roster'] = collector.get_gsw_roster()
    
    gsw_roster = st.session_state['gsw_roster']
    
    if gsw_roster:
        player_names = [player['full_name'] for player in gsw_roster]
        
        selected_player_name = st.sidebar.selectbox(
            "Selecione um jogador do GSW:",
            player_names
        )
        
        if st.sidebar.button("Carregar Jogador"):
            with st.spinner(f"Carregando dados de {selected_player_name}..."):
                selected_player = next((p for p in gsw_roster if p['full_name'] == selected_player_name), None)
                
                if selected_player:
                    st.session_state['selected_entity'] = selected_player
                    st.session_state['entity_type'] = 'player'
                    st.session_state["last_entity_type"] = "player"
                    st.sidebar.markdown(
                        f'<div class="info-box info-green"><i class="fas fa-check-circle"></i>Jogador carregado: {selected_player["full_name"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.sidebar.markdown(
                        f'<div style="padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #721c24; margin-right: 8px;"></i>Erro ao carregar jogador!</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.sidebar.markdown(
            '<div style="padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #721c24; margin-right: 8px;"></i>Erro ao carregar roster do GSW!</div>',
            unsafe_allow_html=True,
        )

if 'selected_entity' in st.session_state:
    entity = st.session_state['selected_entity']
    entity_type = st.session_state['entity_type']

    st.markdown(f'<h2></i>Análise: {entity["full_name"]}</h2>', unsafe_allow_html=True)

    if 'game_log' not in st.session_state or st.session_state.get('last_entity_id') != entity['id']:
        with st.spinner("Carregando dados da NBA..."):
            try:
                if entity_type == 'player':
                    df = collector.get_player_game_log(entity['id'])
                else:
                    df = collector.get_team_game_log(entity['id'])

                if df is not None:
                    if len(df) > 0:
                        st.session_state['game_log'] = df
                        st.session_state['last_entity_id'] = entity['id']
                        st.session_state['features_df'] = None  # Resetar features quando mudar de entidade
                        st.markdown(f'<div class="info-box info-green"><i class="fas fa-check-circle"></i>Dados carregados com sucesso! {len(df)} jogos encontrados.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div style="padding: 0.5rem; background-color: #fff3cd; border-radius: 0.25rem;"><i class="fas fa-exclamation-triangle" style="color: #856404; margin-right: 8px;"></i>Dados carregados, mas nenhum jogo encontrado para {entity["full_name"]} na temporada {season}.</div>', unsafe_allow_html=True)
                        st.info(f"💡 **Possíveis causas:**\n- A temporada {season} pode ainda não ter começado ou não ter dados disponíveis na API\n- Tente usar uma temporada anterior (ex: 2023-24)\n- Verifique se o {'jogador' if entity_type == 'player' else 'time'} jogou na temporada {season}\n- A API da NBA pode ter atraso na disponibilização dos dados")
                        st.session_state['game_log'] = df  # Salvar mesmo se vazio para debug
                else:
                    st.markdown(f'<div style="padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #721c24; margin-right: 8px;"></i>Não foi possível carregar os dados. Verifique se a temporada {season} está disponível para {entity["full_name"]}.</div>', unsafe_allow_html=True)
                    st.info(f"💡 **Dicas:**\n- Verifique se a temporada {season} existe e tem dados disponíveis\n- Tente usar uma temporada anterior (ex: 2023-24)\n- Verifique sua conexão com a internet\n- Verifique os logs no console para mais detalhes")
            except Exception as e:
                st.markdown(f'<div style="padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #721c24; margin-right: 8px;"></i>Erro ao carregar dados: {str(e)}</div>', unsafe_allow_html=True)
                with st.expander("🔍 Ver detalhes do erro"):
                    st.exception(e)

    if 'game_log' in st.session_state:
        df = st.session_state['game_log']

        if df is not None and len(df) > 0:
            st.markdown(f'<div class="info-box info-green"><i class="fas fa-check-circle"></i>{len(df)} jogos carregados para {entity["full_name"]} na temporada {season}</div>', unsafe_allow_html=True)
        elif df is not None:
            st.markdown(f'<div style="padding: 0.5rem; background-color: #fff3cd; border-radius: 0.25rem;"><i class="fas fa-exclamation-triangle" style="color: #856404; margin-right: 8px;"></i>Dados carregados, mas nenhum jogo encontrado para {entity["full_name"]} na temporada {season}</div>', unsafe_allow_html=True)

        if st.button("Criar Features"):
            with st.spinner("Criando features..."):
                if entity_type == 'player':
                    features_df = feature_engineer.create_player_features(df)
                else:
                    features_df = feature_engineer.create_team_features(df)

                if features_df is not None and not features_df.empty:
                    st.session_state['features_df'] = features_df
                    st.markdown(f'<div class="info-box info-green"><i class="fas fa-check-circle"></i>Features criadas! {len(features_df)} amostras disponíveis.</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div style="padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #721c24; margin-right: 8px;"></i>Não foi possível criar features. Dados insuficientes.</div>', unsafe_allow_html=True)

        if df is not None and len(df) > 0:
            st.markdown("---")

            tab1, tab2, tab3 = st.tabs(["Regressão Linear", "Regressão Logística", "Dados"])

            if 'features_df' in st.session_state and st.session_state['features_df'] is not None:
                features_df = st.session_state['features_df']
            else:
                features_df = None

            with tab1:
                st.subheader("Regressão Linear")

                if features_df is not None and len(features_df) > 0:
                    target_var = st.selectbox(
                        "Variável Dependente (Y):",
                        ["target_pts", "target_reb", "target_ast"],
                        format_func=lambda x: {
                            "target_pts": "Pontos",
                            "target_reb": "Rebotes",
                            "target_ast": "Assistências"
                        }[x]
                    )

                    available_features = feature_engineer.get_available_features(entity_type)

                    selected_features = st.multiselect(
                        "Variáveis Independentes (X):",
                        available_features,
                        default=available_features[:5] if len(available_features) >= 5 else available_features
                    )

                    if st.button("Treinar Modelo de Regressão Linear", type="primary"):
                        if len(selected_features) == 0:
                            st.markdown('<div style="padding: 0.5rem; background-color: #fff3cd; border-radius: 0.25rem;"><i class="fas fa-exclamation-triangle" style="color: #856404; margin-right: 8px;"></i>Selecione pelo menos uma variável independente!</div>', unsafe_allow_html=True)
                        else:
                            with st.spinner("Treinando modelo..."):
                                X = features_df[selected_features].copy()
                                y = features_df[target_var].copy()

                                mask = ~(X.isnull().any(axis=1) | y.isnull())
                                X = X[mask]
                                y = y[mask]

                                if len(X) < 2:
                                    st.markdown('<div style="padding: 0.5rem; background-color: #2196F3; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #721c24; margin-right: 8px;"></i>Dados insuficientes após limpeza!</div>', unsafe_allow_html=True)
                                else:
                                    from sklearn.model_selection import train_test_split
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y, test_size=0.2, random_state=42
                                    )

                                    model = LinearRegressionModel()
                                    model.fit(X_train, y_train, scale_features=True)

                                    train_results = model.evaluate(X_train, y_train)
                                    test_results = model.evaluate(X_test, y_test)

                                    st.session_state['linear_model'] = model
                                    st.session_state['linear_X_train'] = X_train
                                    st.session_state['linear_y_train'] = y_train
                                    st.session_state['linear_X_test'] = X_test
                                    st.session_state['linear_y_test'] = y_test

                                    col1, col2, col3, col4 = st.columns(4)

                                    with col1:
                                        st.metric("R² Score (Teste)", f"{test_results['r2']:.4f}")

                                    with col2:
                                        st.metric("RMSE (Teste)", f"{test_results['rmse']:.2f}")

                                    with col3:
                                        st.metric("MAE (Teste)", f"{test_results['mae']:.2f}")

                                    with col4:
                                        st.metric("MSE (Teste)", f"{test_results['mse']:.2f}")

                                    st.subheader("Coeficientes do Modelo")
                                    coef = model.get_coefficients()

                                    coef_df = pd.DataFrame({
                                        'Variável': list(coef.keys()),
                                        'Coeficiente': list(coef.values())
                                    })
                                    st.dataframe(coef_df, width='stretch')
                                    
                                    st.subheader("Previsões vs Valores Reais")
                                    
                                    comparison_df = pd.DataFrame({
                                        'Valor Real': y_test.values,
                                        'Valor Previsto': test_results['predictions'],
                                        'Erro': y_test.values - test_results['predictions'],
                                        'Erro %': ((y_test.values - test_results['predictions']) / y_test.values * 100).round(2)
                                    })
                                    comparison_df.index = range(1, len(comparison_df) + 1)
                                    comparison_df.index.name = 'Jogo'
                                    
                                    st.dataframe(comparison_df.head(10), width='stretch')
                                    
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Erro Médio", f"{comparison_df['Erro'].mean():.2f}")
                                    with col2:
                                        st.metric("Erro Mínimo", f"{comparison_df['Erro'].min():.2f}")
                                    with col3:
                                        st.metric("Erro Máximo", f"{comparison_df['Erro'].max():.2f}")

                                    st.subheader("Visualizações")

                                    visualizer = RegressionVisualizer()

                                    fig1 = visualizer.plot_scatter_with_regression(
                                        X_test.iloc[:, 0] if len(selected_features) > 0 else X_test,
                                        y_test,
                                        test_results['predictions'],
                                        title="Diagrama de Dispersão com Linha de Regressão"
                                    )
                                    st.pyplot(fig1)

                                    fig2 = visualizer.plot_prediction_vs_reality(
                                        y_test,
                                        test_results['predictions'],
                                        title="Previsão vs. Realidade"
                                    )
                                    st.pyplot(fig2)

                                    fig3 = visualizer.plot_trend_with_confidence(
                                        X_test.iloc[:, 0] if len(selected_features) > 0 else X_test,
                                        y_test,
                                        test_results['predictions'],
                                        model,
                                        title="Tendência com Intervalo de Confiança"
                                    )
                                    st.pyplot(fig3)

                else:
                    st.info("( ℹ ) Crie as features primeiro clicando em 'Criar Features' para poder treinar modelos de regressão logística.")

            with tab2:
                st.subheader("Regressão Logística")

                if features_df is not None and len(features_df) > 0:
                    st.info("Para regressão logística, você pode usar variáveis binárias (como Vitória/Derrota) ou converter variáveis contínuas em binárias usando um threshold.")

                    target_var_log = st.selectbox(
                        "Variável Dependente (Y):",
                        ["target_victory", "target_pts", "target_reb", "target_ast"],
                        format_func=lambda x: {
                            "target_victory": "Vitória/Derrota (1/0)",
                            "target_pts": "Pontos",
                            "target_reb": "Rebotes",
                            "target_ast": "Assistências"
                        }[x],
                        key="log_target"
                    )

                    available_features_log = feature_engineer.get_available_features(entity_type)

                    selected_features_log = st.multiselect(
                        "Variáveis Independentes (X):",
                        available_features_log,
                        default=available_features_log[:5] if len(available_features_log) >= 5 else available_features_log,
                        key="log_features"
                    )

                    threshold = None
                    if target_var_log != "target_victory":
                        target_max = float(features_df[target_var_log].max())
                        target_min = float(features_df[target_var_log].min())
                        target_median = float(features_df[target_var_log].median())

                        max_threshold = max(100.0, target_max * 1.1) if target_max > 0 else 100.0

                        threshold = st.number_input(
                            "Threshold para classificação binária:",
                            min_value=target_min,
                            max_value=max_threshold,
                            value=target_median,
                            step=1.0,
                            help=f"Valores disponíveis: min={target_min:.1f}, max={target_max:.1f}, mediana={target_median:.1f}"
                        )
                    else:
                        st.info("( ℹ ) Usando variável binária de Vitória/Derrota. Vitória = 1, Derrota = 0")

                    if st.button("Treinar Modelo de Regressão Logística", type="primary"):
                        if len(selected_features_log) == 0:
                            st.markdown(
                                '<div style="padding: 0.5rem; background-color: #fff3cd; border-radius: 0.25rem;"><i class="fas fa-exclamation-triangle" style="color: #856404; margin-right: 8px;"></i>Selecione pelo menos uma variável independente!</div>',
                                unsafe_allow_html=True,
                            )
                        else:
                            with st.spinner("Treinando modelo..."):
                                X = features_df[selected_features_log].copy()
                                y = features_df[target_var_log].copy()

                                if target_var_log == "target_victory":
                                    y_binary = y.astype(int)  # Já é binário
                                else:
                                    y_binary = (y > threshold).astype(int)

                                mask = ~(X.isnull().any(axis=1) | y_binary.isnull())
                                X = X[mask]
                                y_binary = y_binary[mask]

                                if len(X) < 2:
                                    st.markdown('<div style="padding: 0.5rem; background-color: #f8d7da; border-radius: 0.25rem;"><i class="fas fa-times-circle" style="color: #2196F3; margin-right: 8px;"></i>Dados insuficientes após limpeza!</div>', unsafe_allow_html=True)
                                else:
                                    from sklearn.model_selection import train_test_split
                                    X_train, X_test, y_train, y_test = train_test_split(
                                        X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
                                    )

                                    model = LogisticRegressionModel()
                                    model.fit(X_train, y_train, scale_features=True)

                                    train_results = model.evaluate(X_train, y_train)
                                    test_results = model.evaluate(X_test, y_test)

                                    st.session_state['logistic_model'] = model
                                    st.session_state['logistic_X_test'] = X_test
                                    st.session_state['logistic_y_test'] = y_test

                                    col1, col2, col3 = st.columns(3)

                                    with col1:
                                        st.metric("Accuracy (Teste)", f"{test_results['accuracy']:.4f}")

                                    with col2:
                                        if threshold is not None:
                                            st.metric("Threshold", f"{threshold:.1f}")
                                        else:
                                            st.metric("Tipo", "Binário (Vitória/Derrota)")

                                    with col3:
                                        if target_var_log == "target_victory":
                                            prob_victory = test_results['probabilities'][:, 1].mean()
                                            st.metric("Prob. Média Vitória", f"{prob_victory:.2%}")
                                        else:
                                            st.metric("Amostras", f"{len(y_test)}")

                                    if target_var_log == "target_victory":
                                        st.markdown('<h3><i class="fas fa-chart-pie" style="color: #2196F3; margin-right: 8px;"></i>Análise de Probabilidades</h3>', unsafe_allow_html=True)
                                        prob_test = test_results['probabilities'][:, 1]

                                        prob_df = pd.DataFrame({
                                            'Probabilidade Vitória': prob_test,
                                            'Previsão': test_results['predictions'],
                                            'Real': y_test.values
                                        })
                                        prob_df['Resultado'] = prob_df['Real'].map({1: 'Vitória', 0: 'Derrota'})
                                        prob_df = prob_df.sort_values('Probabilidade Vitória', ascending=False)

                                        st.write("**Exemplos de Probabilidades Previstas:**")
                                        st.dataframe(prob_df.head(10)[['Probabilidade Vitória', 'Previsão', 'Resultado']], width='stretch')

                                        st.markdown('<div class="info-box info-green"><i class="fas fa-lightbulb"></i><strong>Interpretação:</strong> O modelo prevê a probabilidade de vitória para cada jogo. Valores &gt; 0.5 indicam previsão de vitória, valores &lt; 0.5 indicam previsão de derrota.</div>', unsafe_allow_html=True)

                                    st.subheader("Coeficientes do Modelo")
                                    coef = model.get_coefficients()

                                    coef_df = pd.DataFrame({
                                        'Variável': list(coef.keys()),
                                        'Coeficiente': list(coef.values())
                                    })
                                    st.dataframe(coef_df, width='stretch')

                                    st.subheader("Visualizações")
                                    visualizer = RegressionVisualizer()

                                    st.write("**1. Matriz de Confusão**")
                                    fig1 = visualizer.plot_confusion_matrix(
                                        y_test,
                                        test_results['predictions'],
                                        title="Matriz de Confusão"
                                    )
                                    st.pyplot(fig1)

                                    st.write("**2. Curva ROC (Receiver Operating Characteristic)**")
                                    fig2 = visualizer.plot_roc_curve(
                                        y_test,
                                        test_results['probabilities'],
                                        title="Curva ROC"
                                    )
                                    st.pyplot(fig2)

                                    st.write("**3. Gráfico de Probabilidades Previstas**")
                                    fig3 = visualizer.plot_predicted_probabilities(
                                        y_test,
                                        test_results['probabilities'],
                                        title="Gráfico de Probabilidades Previstas"
                                    )
                                    st.pyplot(fig3)

                                    st.write("**4. Gráfico de Importância de Variáveis**")
                                    fig4 = visualizer.plot_feature_importance(
                                        model.model,
                                        selected_features_log,
                                        title="Importância de Variáveis"
                                    )
                                    st.pyplot(fig4)

                                    st.write("**5. Diagrama de Dispersão**")
                                    fig5 = visualizer.plot_scatter_with_regression(
                                        X_test.iloc[:, 0] if len(selected_features_log) > 0 else X_test,
                                        y_test,
                                        test_results['predictions'],
                                        title="Diagrama de Dispersão com Linha de Regressão"
                                    )
                                    st.pyplot(fig5)

                                    st.write("**6. Tendência com Intervalo de Confiança**")
                                    fig6 = visualizer.plot_trend_with_confidence(
                                        X_test.iloc[:, 0] if len(selected_features_log) > 0 else X_test,
                                        y_test,
                                        test_results['predictions'],
                                        model.model,
                                        title="Tendência com Intervalo de Confiança"
                                    )
                                    st.pyplot(fig6)

                                    st.subheader("Relatório de Classificação")
                                    report_df = pd.DataFrame(test_results['classification_report']).transpose()
                                    st.dataframe(report_df, width='stretch')
                else:
                    st.info("( ℹ )  Crie as features primeiro clicando em 'Criar Features' para poder treinar modelos de regressão logística.")

            with tab3:
                st.markdown('<h3><i class="fas fa-database sidebar-icon" style="color: #2196F3;"></i>Dados</h3>', unsafe_allow_html=True)

                if df is not None and len(df) > 0:
                    st.write("**Game Log:**")
                    st.dataframe(df, width='stretch')
                    st.markdown(f'<div class="info-box info-blue"><i class="fas fa-chart-bar"></i>Total de jogos: {len(df)}</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="info-box info-yellow"><i class="fas fa-exclamation-triangle"></i>Nenhum dado de jogo disponível.</div>', unsafe_allow_html=True)

                if features_df is not None and len(features_df) > 0:
                    st.write("**Features Criadas:**")
                    st.dataframe(features_df, width='stretch')

                    st.write("**Estatísticas Descritivas:**")
                    st.dataframe(features_df.describe(), width='stretch')
                else:
                    st.markdown('<div class="info-box info-blue"><i class="fas fa-info-circle"></i>Features ainda não foram criadas. Clique em "Criar Features" para gerar.</div>', unsafe_allow_html=True)

else:
    st.markdown(
        '<div class="info-box info-red"><i class="fas fa-exclamation-circle"></i>Erro ao carregar Golden State Warriors automaticamente. Recarregue a página.</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")
st.markdown('<div style="text-align: center; color: gray;">Atividade de Redes Neurais Artificiais - Regressão Linear e Logística</div>', unsafe_allow_html=True)
