"""
GSW MLP Classifier

Treina uma MLP (Adam optimizer) para prever vitória/derrota do GSW.

Executar:
    streamlit run Atividade_nba/gsw_mlp_app.py
"""

import os
import sys
import random
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict

warnings.filterwarnings("ignore")

# Importar módulos locais
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

try:
    from feature_engineering import FeatureEngineer
    from regression_models import LinearRegressionModel, LogisticRegressionModel
except ImportError:
    st.error("Erro ao importar módulos locais")
    st.stop()

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    st.error("TensorFlow não instalado. Execute: pip install tensorflow")
    st.stop()


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_team_data(season: str):
    """Carrega dados reais do time via API."""
    try:
        from data_collector import NBADataCollector
        collector = NBADataCollector(season=season)
        return collector.get_team_game_log(1610612744)  # GSW team ID
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return None


@dataclass
class MLPConfig:
    hidden_layers: List[int]
    activation: str = "relu"
    dropout: float = 0.0
    epochs: int = 200
    batch_size: int = 16
    validation_split: float = 0.2
    patience: int = 20
    learning_rate: float = 0.001
    use_batch_norm: bool = False


def build_mlp(input_dim: int, cfg: MLPConfig) -> keras.Model:
    """Constrói modelo MLP com BatchNormalization opcional."""
    model = keras.Sequential([layers.Input(shape=(input_dim,))])
    
    for i, units in enumerate(cfg.hidden_layers):
        # Inicialização He para ReLU
        if cfg.activation == "relu":
            kernel_init = keras.initializers.HeNormal()
        else:
            kernel_init = keras.initializers.GlorotUniform()
        
        model.add(layers.Dense(
            units, 
            activation=cfg.activation,
            kernel_initializer=kernel_init,
            kernel_regularizer=keras.regularizers.l2(0.001)
        ))
        
        if cfg.use_batch_norm:
            model.add(layers.BatchNormalization())
        
        if cfg.dropout > 0:
            model.add(layers.Dropout(cfg.dropout))
    
    # Camada de saída
    model.add(layers.Dense(1, activation="sigmoid"))
    return model


def train_mlp(X: np.ndarray, y: np.ndarray, cfg: MLPConfig):
    """Treina MLP com Adam optimizer e callbacks avançados."""
    set_seed(42)
    
    model = build_mlp(X.shape[1], cfg)
    
    # Optimizer com clipnorm para estabilidade
    optimizer = keras.optimizers.Adam(
        learning_rate=cfg.learning_rate,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=cfg.patience,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=cfg.patience // 2,
        min_lr=1e-6,
        verbose=0
    )
    
    history = model.fit(
        X, y,
        validation_split=cfg.validation_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    return model, history


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calcula métricas de avaliação."""
    y_true = y_true.reshape(-1)
    y_pred = y_pred.reshape(-1)
    
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # R²
    var_y = float(np.var(y_true))
    if var_y < 1e-12:
        r2 = 0.0
    else:
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
    
    # Accuracy (threshold 0.5)
    accuracy = float(np.mean((y_pred >= 0.5).astype(int) == y_true))
    
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R²": r2,
        "Accuracy": accuracy
    }


def plot_training_history(history):
    """Plota curvas de loss e accuracy."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss
    ax1.plot(history.history.get("loss", []), label="Treino", linewidth=2)
    ax1.plot(history.history.get("val_loss", []), label="Validação", linestyle="--", linewidth=2)
    ax1.set_title("Loss durante Treinamento")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(history.history.get("accuracy", []), label="Treino", linewidth=2)
    ax2.plot(history.history.get("val_accuracy", []), label="Validação", linestyle="--", linewidth=2)
    ax2.set_title("Accuracy durante Treinamento")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """Plota previsões vs realidade."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histograma de erros
    errors = y_true - y_pred
    axes[0].hist(errors, bins=20, color="#1976d2", alpha=0.7, edgecolor="black")
    axes[0].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[0].set_title("Distribuição dos Erros")
    axes[0].set_xlabel("Erro (Real - Previsto)")
    axes[0].set_ylabel("Frequência")
    axes[0].grid(alpha=0.3)
    
    # Scatter Real vs Previsto
    axes[1].scatter(y_true, y_pred, alpha=0.6, s=50)
    axes[1].plot([0, 1], [0, 1], "r--", linewidth=2, label="Ideal")
    axes[1].set_title("Real vs Previsto")
    axes[1].set_xlabel("Valor Real (0 ou 1)")
    axes[1].set_ylabel("Probabilidade Prevista")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim(-0.1, 1.1)
    axes[1].set_ylim(-0.1, 1.1)
    
    # Série temporal de erros
    axes[2].plot(errors, marker="o", markersize=4, alpha=0.6)
    axes[2].axhline(0, color="red", linestyle="--", linewidth=2)
    axes[2].set_title("Erro por Amostra (Teste)")
    axes[2].set_xlabel("Índice da Amostra")
    axes[2].set_ylabel("Erro")
    axes[2].grid(alpha=0.3)
    
    fig.tight_layout()
    return fig


def plot_game_predictions(y_true: np.ndarray, y_pred: np.ndarray):
    """Plota probabilidades por jogo."""
    fig, ax = plt.subplots(figsize=(12, 4))
    
    indices = range(len(y_true))
    ax.plot(indices, y_pred, label="Probabilidade Prevista", linewidth=2, color="#1976d2")
    ax.scatter(indices, y_true, label="Resultado Real", marker="x", s=100, color="black", linewidths=2)
    
    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.5, label="Threshold (0.5)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Previsões por Jogo (Conjunto de Teste)")
    ax.set_xlabel("Jogo")
    ax.set_ylabel("Probabilidade de Vitória")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.tight_layout()
    return fig


# ========================== STREAMLIT APP ==========================

st.set_page_config(page_title="GSW MLP", page_icon="🏀", layout="wide")
st.title("🏀 Golden State Warriors - MLP Classifier")
st.caption("Previsão de vitória/derrota usando MLP (Adam Optimizer)")

# Sidebar - Configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    st.subheader("Dados")
    season = st.selectbox("Temporada", ["2024-25", "2023-24", "2022-23"], index=1)
    
    st.subheader("Arquitetura da Rede")
    hidden_1 = st.number_input("Neurônios - Camada 1", min_value=8, max_value=256, value=24, step=4)
    hidden_2 = st.number_input("Neurônios - Camada 2", min_value=0, max_value=256, value=12, step=4)
    hidden_3 = st.number_input("Neurônios - Camada 3 (0=desativa)", min_value=0, max_value=128, value=0, step=4)
    activation = st.selectbox("Função de Ativação", ["relu", "tanh", "sigmoid"], index=1)  # tanh
    use_batch_norm = st.checkbox("Batch Normalization", value=True)
    dropout = st.slider("Taxa de Dropout", 0.0, 0.5, 0.30, 0.05)
    
    st.subheader("Treinamento")
    learning_rate = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    epochs = st.slider("Épocas Máximas", 50, 500, 430, 10)
    batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=2)  # 32
    patience = st.slider("Early Stopping (Paciência)", 10, 50, 30, 2)
    val_split = st.slider("Validação Split", 0.1, 0.3, 0.25, 0.05)
    
    st.markdown("---")
    train_button = st.button("🚀 Treinar MLP", type="primary")

# Carregar dados
with st.spinner("📊 Carregando dados do GSW..."):
    team_log = load_team_data(season)
    
    if team_log is None or len(team_log) < 10:
        st.error("❌ Dados insuficientes (mínimo 10 jogos). Verifique a conexão com a API.")
        st.stop()

st.success(f"✅ {len(team_log)} jogos carregados")

# Feature Engineering
with st.spinner("🔧 Processando features..."):
    fe = FeatureEngineer()
    team_features = fe.create_team_features(team_log)
    
    if team_features is None or team_features.empty:
        st.error("❌ Erro ao criar features")
        st.stop()

# Selecionar features
selected_features = [
    "avg_pts", "avg_reb", "avg_ast", "avg_fg_pct", "avg_fg3m",
    "pts_last_5", "reb_last_5", "ast_last_5", "std_pts", "prev_pts"
]
selected_features = [f for f in selected_features if f in team_features.columns]

X_all = team_features[selected_features].fillna(team_features[selected_features].median())
y_all = team_features["target_victory"].astype(int)

# Normalizar
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all.values)

# Split treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_all.values,
    test_size=0.2,
    random_state=42,
    stratify=y_all.values
)

# Calcular balanceamento de classes
n_victories = sum(y_all)
n_defeats = len(y_all) - n_victories
class_balance = n_victories / len(y_all)

# Mostrar info dos dados
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📊 Total de Amostras", len(X_all))
with col2:
    st.metric("🎯 Features", len(selected_features))
with col3:
    st.metric("✅ Vitórias / ❌ Derrotas", f"{n_victories} / {n_defeats}")
with col4:
    balance_emoji = "⚖️" if 0.4 <= class_balance <= 0.6 else "⚠️"
    st.metric(f"{balance_emoji} Balanceamento", f"{class_balance:.1%}")

with st.expander("🔍 Ver Features Selecionadas"):
    st.write(pd.DataFrame({"Feature": selected_features}))

# Treinar modelo
if train_button:
    st.markdown("---")
    st.subheader("🤖 Treinamento do Modelo")
    
    # Configurar MLP
    hidden_layers = [hidden_1]
    if hidden_2 > 0:
        hidden_layers.append(hidden_2)
    if hidden_3 > 0:
        hidden_layers.append(hidden_3)
    
    cfg = MLPConfig(
        hidden_layers=hidden_layers,
        activation=activation,
        dropout=dropout,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        patience=patience,
        learning_rate=learning_rate,
        use_batch_norm=use_batch_norm
    )
    
    # Mostrar arquitetura
    st.info(f"🏗️ Arquitetura: {' → '.join(map(str, hidden_layers))} → 1 (Sigmoid)")
    
    # Treinar
    with st.spinner("⏳ Treinando MLP..."):
        model, history = train_mlp(X_train, y_train, cfg)
        proba = model.predict(X_test, verbose=0).reshape(-1)
        
        # Também prever no treino para diagnóstico
        proba_train = model.predict(X_train, verbose=0).reshape(-1)
    
    # Info do treinamento
    final_epoch = len(history.history['loss'])
    final_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    st.success(f"✅ Treinamento concluído em {final_epoch} épocas! Loss final: {final_loss:.4f} | Val Loss: {final_val_loss:.4f}")
    
    # Métricas
    st.subheader("📈 Métricas de Avaliação")
    
    # Métricas de treino e teste
    metrics_train = calculate_metrics(y_train, proba_train)
    metrics = calculate_metrics(y_test, proba)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("MAE (Teste)", f"{metrics['MAE']:.4f}", delta=f"{metrics['MAE']-metrics_train['MAE']:.4f}")
    with col2:
        st.metric("RMSE (Teste)", f"{metrics['RMSE']:.4f}", delta=f"{metrics['RMSE']-metrics_train['RMSE']:.4f}")
    with col3:
        st.metric("R² (Teste)", f"{metrics['R²']:.4f}", delta=f"{metrics['R²']-metrics_train['R²']:.4f}")
    with col4:
        st.metric("Accuracy (Teste)", f"{metrics['Accuracy']:.2%}", delta=f"{metrics['Accuracy']-metrics_train['Accuracy']:.1%}")
    
    # Mostrar métricas de treino em expander
    with st.expander("📊 Comparação Treino vs Teste (detectar overfitting)"):
        comp_df = pd.DataFrame({
            "Conjunto": ["Treino", "Teste"],
            "MAE": [metrics_train['MAE'], metrics['MAE']],
            "RMSE": [metrics_train['RMSE'], metrics['RMSE']],
            "R²": [metrics_train['R²'], metrics['R²']],
            "Accuracy": [metrics_train['Accuracy'], metrics['Accuracy']]
        })
        st.dataframe(comp_df.style.format({
            "MAE": "{:.4f}",
            "RMSE": "{:.4f}",
            "R²": "{:.4f}",
            "Accuracy": "{:.2%}"
        }), use_container_width=True)
    
    # Gráficos de treinamento
    st.subheader("📊 Evolução do Treinamento")
    fig_hist = plot_training_history(history)
    st.pyplot(fig_hist, use_container_width=True)
    
    # Análise de erros
    st.subheader("🔍 Análise de Previsões")
    fig_pred = plot_predictions(y_test, proba)
    st.pyplot(fig_pred, use_container_width=True)
    
    # Previsões por jogo
    st.subheader("🎯 Previsões por Jogo")
    fig_games = plot_game_predictions(y_test, proba)
    st.pyplot(fig_games, use_container_width=True)
    
    # Tabela de previsões
    with st.expander("📋 Ver Tabela Detalhada de Previsões"):
        pred_df = pd.DataFrame({
            "Jogo": range(1, len(y_test) + 1),
            "Real": y_test,
            "Probabilidade": proba,
            "Previsto": (proba >= 0.5).astype(int),
            "Correto": (((proba >= 0.5).astype(int)) == y_test)
        })
        pred_df["Real"] = pred_df["Real"].map({0: "❌ Derrota", 1: "✅ Vitória"})
        pred_df["Previsto"] = pred_df["Previsto"].map({0: "❌ Derrota", 1: "✅ Vitória"})
        pred_df["Correto"] = pred_df["Correto"].map({True: "✅", False: "❌"})
        st.dataframe(pred_df, use_container_width=True)
    
    # ==================== COMPARAÇÃO COM REGRESSÕES ====================
    st.markdown("---")
    st.header("📊 Comparação: MLP vs Regressões")
    st.caption("Comparando previsões (classe binária) da MLP com Regressão Logística e Linear")
    
    with st.spinner("Treinando modelos de regressão para comparação..."):
        # Preparar dados em DataFrame para os modelos de regressão
        X_train_df = pd.DataFrame(X_train, columns=selected_features)
        X_test_df = pd.DataFrame(X_test, columns=selected_features)
        y_train_series = pd.Series(y_train)
        y_test_series = pd.Series(y_test)
        
        # Treinar Regressão Logística
        log_model = LogisticRegressionModel()
        log_model.fit(X_train_df, y_train_series, scale_features=False)  # já escalado
        log_eval = log_model.evaluate(X_test_df, y_test_series)
        log_pred = (log_eval["probabilities"][:, 1] >= 0.5).astype(int)  # Previsões binárias
        log_acc = log_eval["accuracy"]
        
        # Treinar Regressão Linear (prever 0/1 como valores contínuos)
        lin_model = LinearRegressionModel()
        lin_model.fit(X_train_df, y_train_series, scale_features=False)  # já escalado
        lin_pred_raw = lin_model.predict(X_test_df).reshape(-1)
        lin_pred = (lin_pred_raw >= 0.5).astype(int)  # Binarizar com threshold 0.5
        lin_acc = float(np.mean(lin_pred == y_test))
        
        # Previsões da MLP (binarizadas)
        mlp_pred = (proba >= 0.5).astype(int)
        mlp_acc = float(np.mean(mlp_pred == y_test))
    
    # Tabela comparativa de acurácia
    st.subheader("🎯 Acurácia dos Modelos")
    comparison_acc = pd.DataFrame({
        "Modelo": ["Regressão Linear", "Regressão Logística", "MLP (Adam)"],
        "Acurácia": [lin_acc, log_acc, mlp_acc],
        "Acertos": [int(lin_acc * len(y_test)), int(log_acc * len(y_test)), int(mlp_acc * len(y_test))],
        "Erros": [len(y_test) - int(lin_acc * len(y_test)), 
                  len(y_test) - int(log_acc * len(y_test)), 
                  len(y_test) - int(mlp_acc * len(y_test))]
    })
    
    # Destacar o melhor
    best_idx = comparison_acc["Acurácia"].idxmax()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        emoji = "🥇" if best_idx == 0 else ""
        st.metric(f"{emoji} Regressão Linear", f"{lin_acc:.2%}")
    with col2:
        emoji = "🥇" if best_idx == 1 else ""
        st.metric(f"{emoji} Regressão Logística", f"{log_acc:.2%}")
    with col3:
        emoji = "🥇" if best_idx == 2 else ""
        st.metric(f"{emoji} MLP (Adam)", f"{mlp_acc:.2%}")
    
    st.dataframe(comparison_acc.style.format({"Acurácia": "{:.2%}"}), use_container_width=True)
    
    # Gráfico de barras comparativo
    fig_comp, ax_comp = plt.subplots(figsize=(10, 5))
    bars = ax_comp.bar(comparison_acc["Modelo"], comparison_acc["Acurácia"], 
                       color=["#FF9800", "#2196F3", "#4CAF50"], alpha=0.8)
    ax_comp.set_ylim(0, 1)
    ax_comp.set_ylabel("Acurácia", fontsize=12)
    ax_comp.set_title("Comparação de Acurácia: Linear vs Logística vs MLP", fontsize=14, fontweight="bold")
    ax_comp.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax_comp.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    st.pyplot(fig_comp, use_container_width=True)
    
    # Tabela detalhada de previsões comparativas
    st.subheader("🔍 Comparação Detalhada das Previsões")
    
    compare_pred_df = pd.DataFrame({
        "Jogo": range(1, len(y_test) + 1),
        "Real": y_test,
        "MLP": mlp_pred,
        "Logística": log_pred,
        "Linear": lin_pred,
        "MLP ✓": (mlp_pred == y_test).astype(int),
        "Log ✓": (log_pred == y_test).astype(int),
        "Lin ✓": (lin_pred == y_test).astype(int),
    })
    
    # Mapear para emojis
    compare_pred_df["Real"] = compare_pred_df["Real"].map({0: "❌", 1: "✅"})
    compare_pred_df["MLP"] = compare_pred_df["MLP"].map({0: "❌", 1: "✅"})
    compare_pred_df["Logística"] = compare_pred_df["Logística"].map({0: "❌", 1: "✅"})
    compare_pred_df["Linear"] = compare_pred_df["Linear"].map({0: "❌", 1: "✅"})
    compare_pred_df["MLP ✓"] = compare_pred_df["MLP ✓"].map({0: "❌", 1: "✅"})
    compare_pred_df["Log ✓"] = compare_pred_df["Log ✓"].map({0: "❌", 1: "✅"})
    compare_pred_df["Lin ✓"] = compare_pred_df["Lin ✓"].map({0: "❌", 1: "✅"})
    
    st.dataframe(compare_pred_df, use_container_width=True)
    
    # Análise de concordância
    st.subheader("🤝 Análise de Concordância")
    all_agree = np.sum((mlp_pred == log_pred) & (log_pred == lin_pred))
    mlp_log_agree = np.sum(mlp_pred == log_pred)
    mlp_lin_agree = np.sum(mlp_pred == lin_pred)
    log_lin_agree = np.sum(log_pred == lin_pred)
    
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("🎯 Todos Concordam", f"{all_agree}/{len(y_test)}")
    with col_b:
        st.metric("MLP ↔ Logística", f"{mlp_log_agree}/{len(y_test)}")
    with col_c:
        st.metric("MLP ↔ Linear", f"{mlp_lin_agree}/{len(y_test)}")
    with col_d:
        st.metric("Logística ↔ Linear", f"{log_lin_agree}/{len(y_test)}")

else:
    st.info("👆 Configure os parâmetros na barra lateral e clique em 'Treinar MLP'")

st.markdown("---")
st.caption("🏀 GSW MLP Classifier (Adam Optimizer)")
