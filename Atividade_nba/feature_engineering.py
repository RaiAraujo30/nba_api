"""
Módulo para criação de features (variáveis independentes)
"""

import pandas as pd
import numpy as np


class FeatureEngineer:
    """Classe para criar features a partir de dados históricos"""
    
    def __init__(self, league_stats_df=None):
        """
        Args:
            league_stats_df: DataFrame com estatísticas de todos os times (opcional)
        """
        self.league_stats_df = league_stats_df
    
    def create_player_features(self, df):
        """
        Cria features para cada jogo de um jogador
        Baseado em dados históricos até o jogo anterior
        """
        if df is None or df.empty:
            return None
        
        features_list = []
        
        if len(df) < 2:
            return None
        
        for i in range(1, len(df)):
            past_games = df.iloc[:i].copy()
            
            current_game = df.iloc[i]
            
            feature_row = {
                'avg_pts': past_games['PTS'].mean(),
                'avg_reb': past_games['REB'].mean(),
                'avg_ast': past_games['AST'].mean(),
                'avg_min': past_games['MIN'].mean(),
                
                'avg_fgm': past_games['FGM'].mean(),
                'avg_fga': past_games['FGA'].mean(),
                'avg_fg_pct': past_games['FG_PCT'].mean(),
                
                'avg_fg3m': past_games['FG3M'].mean(),
                'avg_fg3a': past_games['FG3A'].mean(),
                'avg_fg3_pct': past_games['FG3_PCT'].mean(),
                
                'avg_ftm': past_games['FTM'].mean(),
                'avg_fta': past_games['FTA'].mean(),
                'avg_ft_pct': past_games['FT_PCT'].mean(),
                
                'avg_oreb': past_games['OREB'].mean(),
                'avg_dreb': past_games['DREB'].mean(),
                
                'avg_stl': past_games['STL'].mean(),
                'avg_blk': past_games['BLK'].mean(),
                'avg_tov': past_games['TOV'].mean(),
                'avg_pf': past_games['PF'].mean(),
                'avg_plus_minus': past_games['PLUS_MINUS'].mean(),
                
                'pts_last_5': past_games['PTS'].tail(5).mean() if len(past_games) >= 5 else past_games['PTS'].mean(),
                'reb_last_5': past_games['REB'].tail(5).mean() if len(past_games) >= 5 else past_games['REB'].mean(),
                'ast_last_5': past_games['AST'].tail(5).mean() if len(past_games) >= 5 else past_games['AST'].mean(),
                
                'std_pts': past_games['PTS'].std(),
                'std_reb': past_games['REB'].std(),
                'std_ast': past_games['AST'].std(),
                
                'max_pts': past_games['PTS'].max(),
                'min_pts': past_games['PTS'].min(),
                
                'prev_pts': past_games.iloc[-1]['PTS'],
                'prev_reb': past_games.iloc[-1]['REB'],
                'prev_ast': past_games.iloc[-1]['AST'],
                
                'target_pts': current_game['PTS'],
                'target_reb': current_game['REB'],
                'target_ast': current_game['AST'],
                
                'game_date': current_game['GAME_DATE'],
                'matchup': current_game['MATCHUP'],
                'wl': current_game['WL'],
                
                'target_victory': 1 if current_game['WL'] == 'W' else 0,
            }
            
            features_list.append(feature_row)
        
        return pd.DataFrame(features_list)
    
    def create_team_features(self, df):
        """
        Cria features para cada jogo de um time
        Baseado em dados históricos até o jogo anterior
        """
        if df is None or df.empty:
            return None
        
        features_list = []
        
        if len(df) < 2:
            return None
        
        for i in range(1, len(df)):
            past_games = df.iloc[:i].copy()
            
            current_game = df.iloc[i]
            
            feature_row = {
                'avg_pts': past_games['PTS'].mean(),
                'avg_reb': past_games['REB'].mean(),
                'avg_ast': past_games['AST'].mean(),
                'avg_min': past_games['MIN'].mean(),
                
                'avg_fgm': past_games['FGM'].mean(),
                'avg_fga': past_games['FGA'].mean(),
                'avg_fg_pct': past_games['FG_PCT'].mean(),
                
                'avg_fg3m': past_games['FG3M'].mean(),
                'avg_fg3a': past_games['FG3A'].mean(),
                'avg_fg3_pct': past_games['FG3_PCT'].mean(),
                
                'avg_ftm': past_games['FTM'].mean(),
                'avg_fta': past_games['FTA'].mean(),
                'avg_ft_pct': past_games['FT_PCT'].mean(),
                
                'avg_oreb': past_games['OREB'].mean(),
                'avg_dreb': past_games['DREB'].mean(),
                
                'avg_stl': past_games['STL'].mean(),
                'avg_blk': past_games['BLK'].mean(),
                'avg_tov': past_games['TOV'].mean(),
                'avg_pf': past_games['PF'].mean(),
                
                'win_pct': past_games['W_PCT'].mean() if 'W_PCT' in past_games.columns else 0,
                
                'pts_last_5': past_games['PTS'].tail(5).mean() if len(past_games) >= 5 else past_games['PTS'].mean(),
                'reb_last_5': past_games['REB'].tail(5).mean() if len(past_games) >= 5 else past_games['REB'].mean(),
                'ast_last_5': past_games['AST'].tail(5).mean() if len(past_games) >= 5 else past_games['AST'].mean(),
                
                'std_pts': past_games['PTS'].std(),
                'std_reb': past_games['REB'].std(),
                'std_ast': past_games['AST'].std(),
                
                'prev_pts': past_games.iloc[-1]['PTS'],
                'prev_reb': past_games.iloc[-1]['REB'],
                'prev_ast': past_games.iloc[-1]['AST'],
                
                'target_pts': current_game['PTS'],
                'target_reb': current_game['REB'],
                'target_ast': current_game['AST'],
                
                'game_date': current_game['GAME_DATE'],
                'matchup': current_game['MATCHUP'],
                'wl': current_game['WL'],
                
                'target_victory': 1 if current_game['WL'] == 'W' else 0,
            }
            
            features_list.append(feature_row)
        
        return pd.DataFrame(features_list)
    
    def get_available_features(self, entity_type='player'):
        """
        Retorna lista de features disponíveis para seleção
        """
        base_features = [
            'avg_pts', 'avg_reb', 'avg_ast', 'avg_min',
            'avg_fgm', 'avg_fga', 'avg_fg_pct',
            'avg_fg3m', 'avg_fg3a', 'avg_fg3_pct',
            'avg_ftm', 'avg_fta', 'avg_ft_pct',
            'avg_oreb', 'avg_dreb',
            'avg_stl', 'avg_blk', 'avg_tov', 'avg_pf',
            'pts_last_5', 'reb_last_5', 'ast_last_5',
            'std_pts', 'std_reb', 'std_ast',
            'max_pts', 'min_pts',
            'prev_pts', 'prev_reb', 'prev_ast',
        ]
        
        if entity_type == 'team':
            base_features.append('win_pct')
        
        return base_features
    
    def get_target_variables(self):
        """Retorna lista de variáveis alvo disponíveis"""
        return ['target_pts', 'target_reb', 'target_ast', 'target_victory']

