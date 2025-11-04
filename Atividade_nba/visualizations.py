"""
Módulo para visualizações dos modelos
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Configurar estilo
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)


class RegressionVisualizer:
    """Classe para visualizações de regressão"""
    
    @staticmethod
    def plot_scatter_with_regression(X, y, y_pred, title="Diagrama de Dispersão com Linha de Regressão"):
        """
        Cria diagrama de dispersão com linha de regressão
        
        Args:
            X: Variáveis independentes (usa a primeira se múltipla)
            y: Valores reais
            y_pred: Valores previstos
            title: Título do gráfico
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Se X é múltipla, usar a primeira feature
        if isinstance(X, pd.DataFrame):
            x_values = X.iloc[:, 0].values
            x_label = X.columns[0]
        else:
            x_values = X if isinstance(X, np.ndarray) else X.values
            x_label = "Variável Independente"
        
        # Converter y para array numpy se necessário
        y_values = y.values if isinstance(y, pd.Series) else (y if isinstance(y, np.ndarray) else np.array(y))
        y_pred_values = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        # Plot dos pontos
        ax.scatter(x_values, y_values, alpha=0.6, label='Valores Reais', color='blue')
        
        # Ordenar para linha de regressão
        sorted_indices = np.argsort(x_values)
        x_sorted = x_values[sorted_indices]
        y_pred_sorted = y_pred_values[sorted_indices]
        
        # Linha de regressão
        ax.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='Linha de Regressão', alpha=0.8)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Valores Reais', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_prediction_vs_reality(y_true, y_pred, title="Previsão vs. Realidade"):
        """
        Gráfico comparando valores previstos com valores observados
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            title: Título do gráfico
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Converter para array numpy se necessário
        y_true_values = y_true.values if isinstance(y_true, pd.Series) else (y_true if isinstance(y_true, np.ndarray) else np.array(y_true))
        y_pred_values = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        # Plot dos pontos
        ax.scatter(y_true_values, y_pred_values, alpha=0.6, color='blue')
        
        # Linha de referência perfeita (y = x)
        min_val = min(min(y_true_values), min(y_pred_values))
        max_val = max(max(y_true_values), max(y_pred_values))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                label='Linha Perfeita (y=x)', alpha=0.8)
        
        ax.set_xlabel('Valores Reais', fontsize=12)
        ax.set_ylabel('Valores Previstos', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_trend_with_confidence(X, y, y_pred, model, title="Tendência com Intervalo de Confiança"):
        """
        Gráfico de tendência com intervalo de confiança
        
        Args:
            X: Features
            y: Valores reais
            y_pred: Valores previstos
            model: Modelo treinado (para calcular intervalo de confiança)
            title: Título do gráfico
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Se X é múltipla, usar a primeira feature
        if isinstance(X, pd.DataFrame):
            x_values = X.iloc[:, 0].values
            x_label = X.columns[0]
        else:
            x_values = X if isinstance(X, np.ndarray) else X.values
            x_label = "Variável Independente"
        
        # Ordenar por X
        sorted_indices = np.argsort(x_values)
        x_sorted = x_values[sorted_indices]
        
        # Converter y para array numpy se for Series
        y_values = y.values if isinstance(y, pd.Series) else (y if isinstance(y, np.ndarray) else np.array(y))
        y_sorted = y_values[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices] if isinstance(y_pred, np.ndarray) else np.array(y_pred)[sorted_indices]
        
        # Calcular resíduos para intervalo de confiança
        residuals = y_sorted - y_pred_sorted
        std_residuals = np.std(residuals)
        
        # Intervalo de confiança (95%)
        confidence_interval = 1.96 * std_residuals
        
        # Plot
        ax.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2, label='Previsão', alpha=0.8)
        ax.fill_between(x_sorted, 
                        y_pred_sorted - confidence_interval,
                        y_pred_sorted + confidence_interval,
                        alpha=0.3, color='red', label='Intervalo de Confiança (95%)')
        ax.scatter(x_sorted, y_sorted, alpha=0.5, color='blue', label='Valores Reais', s=30)
        
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel('Valores', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, title="Matriz de Confusão"):
        """
        Matriz de confusão para regressão logística
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            title: Título do gráfico
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Classe 0', 'Classe 1'],
                   yticklabels=['Classe 0', 'Classe 1'])
        
        ax.set_xlabel('Valores Previstos', fontsize=12)
        ax.set_ylabel('Valores Reais', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_residuals(y_true, y_pred, title="Análise de Resíduos"):
        """
        Gráfico de resíduos
        
        Args:
            y_true: Valores reais
            y_pred: Valores previstos
            title: Título do gráfico
        """
        # Converter para array numpy se necessário
        y_true_values = y_true.values if isinstance(y_true, pd.Series) else (y_true if isinstance(y_true, np.ndarray) else np.array(y_true))
        y_pred_values = y_pred if isinstance(y_pred, np.ndarray) else np.array(y_pred)
        
        residuals = y_true_values - y_pred_values
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico de resíduos vs previstos
        ax1.scatter(y_pred_values, residuals, alpha=0.6, color='blue')
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Valores Previstos', fontsize=12)
        ax1.set_ylabel('Resíduos', fontsize=12)
        ax1.set_title('Resíduos vs. Previstos', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Histograma de resíduos
        ax2.hist(residuals, bins=20, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax2.set_xlabel('Resíduos', fontsize=12)
        ax2.set_ylabel('Frequência', fontsize=12)
        ax2.set_title('Distribuição dos Resíduos', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

