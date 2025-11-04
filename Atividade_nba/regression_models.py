"""
Módulo para Regressão Linear e Logística
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, accuracy_score, classification_report
)
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class LinearRegressionModel:
    """Classe para Regressão Linear"""
    
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y, scale_features=True):
        """
        Treina o modelo de regressão linear
        
        Args:
            X: Features (variáveis independentes)
            y: Target (variável dependente)
            scale_features: Se deve normalizar as features
        """
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        X_train = X.copy()
        
        if scale_features:
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        self.model.fit(X_train, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Faz previsões com o modelo"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado ainda. Use fit() primeiro.")
        
        X_test = X.copy()
        
        if self.scaler is not None:
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return self.model.predict(X_test)
    
    def get_coefficients(self):
        """Retorna os coeficientes do modelo"""
        if not self.is_fitted:
            return None
        
        coef_dict = {}
        if self.feature_names:
            for i, feature in enumerate(self.feature_names):
                coef_dict[feature] = self.model.coef_[i]
        else:
            for i in range(len(self.model.coef_)):
                coef_dict[f'feature_{i}'] = self.model.coef_[i]
        
        coef_dict['intercept'] = self.model.intercept_
        
        return coef_dict
    
    def evaluate(self, X, y):
        """Avalia o modelo"""
        y_pred = self.predict(X)
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actual': y.values if isinstance(y, pd.Series) else y
        }
    
    def get_model_info(self):
        """Retorna informações do modelo"""
        if not self.is_fitted:
            return None
        
        coef = self.get_coefficients()
        
        info = {
            'intercept': coef['intercept'],
            'coefficients': {k: v for k, v in coef.items() if k != 'intercept'},
            'n_features': len(self.model.coef_)
        }
        
        return info


class LogisticRegressionModel:
    """Classe para Regressão Logística"""
    
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
    def fit(self, X, y, scale_features=True):
        """
        Treina o modelo de regressão logística
        
        Args:
            X: Features (variáveis independentes)
            y: Target (variável dependente binária)
            scale_features: Se deve normalizar as features
        """
        self.feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else None
        
        X_train = X.copy()
        
        if scale_features:
            X_train = pd.DataFrame(
                self.scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index
            )
        
        self.model.fit(X_train, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """Faz previsões com o modelo"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado ainda. Use fit() primeiro.")
        
        X_test = X.copy()
        
        if self.scaler is not None:
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return self.model.predict(X_test)
    
    def predict_proba(self, X):
        """Retorna probabilidades de cada classe"""
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado ainda. Use fit() primeiro.")
        
        X_test = X.copy()
        
        if self.scaler is not None:
            X_test = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index
            )
        
        return self.model.predict_proba(X_test)
    
    def get_coefficients(self):
        """Retorna os coeficientes do modelo"""
        if not self.is_fitted:
            return None
        
        coef_dict = {}
        if self.feature_names:
            for i, feature in enumerate(self.feature_names):
                coef_dict[feature] = self.model.coef_[0][i]
        else:
            for i in range(len(self.model.coef_[0])):
                coef_dict[f'feature_{i}'] = self.model.coef_[0][i]
        
        coef_dict['intercept'] = self.model.intercept_[0]
        
        return coef_dict
    
    def evaluate(self, X, y):
        """Avalia o modelo"""
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        accuracy = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        
        # Classificação report
        report = classification_report(y, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'predictions': y_pred,
            'probabilities': y_proba,
            'actual': y.values if isinstance(y, pd.Series) else y
        }
    
    def get_model_info(self):
        """Retorna informações do modelo"""
        if not self.is_fitted:
            return None
        
        coef = self.get_coefficients()
        
        info = {
            'intercept': coef['intercept'],
            'coefficients': {k: v for k, v in coef.items() if k != 'intercept'},
            'n_features': len(self.model.coef_[0])
        }
        
        return info

