"""
Módulo para coleta de dados da NBA usando nba_api
"""

import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, teamgamelog, leaguedashteamstats
from nba_api.stats.static import players, teams
from nba_api.stats.library.parameters import Season, SeasonTypeAllStar
import time


class NBADataCollector:
    """Classe para coletar dados da NBA"""
    
    def __init__(self, season='2024-25'):
        self.season = season
        self.season_type = SeasonTypeAllStar.regular  # Usar 'regular' que é herdado de SeasonType
        
    def find_player(self, player_name):
        """Busca um jogador pelo nome"""
        try:
            player_list = players.find_players_by_full_name(player_name)
            if not player_list:
                return None
            return player_list[0]
        except Exception as e:
            print(f"Erro ao buscar jogador: {e}")
            return None
    
    def find_team(self, team_name):
        """Busca um time pelo nome"""
        try:
            # Tentar buscar por nome completo primeiro
            team_list = teams.find_teams_by_full_name(team_name)
            
            # Se não encontrou, tentar por apelido
            if not team_list:
                team_list = teams.find_teams_by_nickname(team_name)
            
            # Se ainda não encontrou, tentar por cidade
            if not team_list:
                team_list = teams.find_teams_by_city(team_name)
            
            # Se ainda não encontrou, tentar por abreviação
            if not team_list:
                # Tentar buscar como abreviação
                team_obj = teams.find_team_by_abbreviation(team_name.upper())
                if team_obj:
                    team_list = [team_obj]
            
            if not team_list:
                return None
            
            # Se encontrou múltiplos, retornar o primeiro
            # Mas se encontrou exatamente um, retornar ele
            if len(team_list) == 1:
                return team_list[0]
            
            # Se encontrou múltiplos, tentar encontrar exata correspondência
            team_name_lower = team_name.lower()
            for team in team_list:
                if team_name_lower in team['full_name'].lower():
                    return team
            
            # Se não encontrou correspondência exata, retornar o primeiro
            return team_list[0]
            
        except Exception as e:
            print(f"Erro ao buscar time: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_player_game_log(self, player_id):
        """Obtém o log de jogos de um jogador"""
        try:
            time.sleep(0.6)  # Rate limiting
            player_log = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=self.season,
                season_type_all_star=self.season_type
            )
            
            df = player_log.player_game_log.get_data_frame()
            
            # Verificar se retornou algo válido
            if df is None:
                return None
            
            # Se retornou um DataFrame vazio, retornar um DataFrame vazio (não None)
            # para que possamos distinguir entre erro e sem dados
            if df.empty:
                return pd.DataFrame()
            
            # Garantir que as colunas numéricas sejam numéricas
            numeric_columns = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 
                               'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                               'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'MIN', 'PLUS_MINUS']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Erro ao obter log de jogos do jogador: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_team_game_log(self, team_id):
        """Obtém o log de jogos de um time"""
        try:
            time.sleep(0.6)  # Rate limiting
            
            # Primeiro tentar usar TeamGameLog
            team_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=self.season,
                season_type_all_star=self.season_type
            )
            
            # Tentar obter DataFrame usando o método normal
            df = None
            try:
                if hasattr(team_log, 'team_game_log') and team_log.team_game_log is not None:
                    df = team_log.team_game_log.get_data_frame()
            except Exception as e:
                print(f"WARNING - Erro ao obter DataFrame via team_game_log: {e}")
            
            # Se TeamGameLog retornou dados, usar
            if df is not None and not df.empty:
                # Garantir que as colunas numéricas sejam numéricas
                numeric_columns = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT',
                                   'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                                   'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'MIN']
                
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                return df
            
            # Se TeamGameLog não funcionou, usar LeagueGameLog como alternativa
            print(f"INFO - TeamGameLog vazio, tentando LeagueGameLog como alternativa...")
            from nba_api.stats.endpoints import leaguegamelog
            from nba_api.stats.library.parameters import PlayerOrTeamAbbreviation
            
            time.sleep(0.6)  # Rate limiting adicional
            
            league_log = leaguegamelog.LeagueGameLog(
                season=self.season,
                season_type_all_star=self.season_type,
                player_or_team_abbreviation=PlayerOrTeamAbbreviation.team  # Buscar dados de times
            )
            
            df_all = league_log.league_game_log.get_data_frame()
            
            if df_all is not None and not df_all.empty:
                # Filtrar pelo team_id
                if 'TEAM_ID' in df_all.columns:
                    df = df_all[df_all['TEAM_ID'] == team_id].copy()
                    
                    if not df.empty:
                        # Mapear colunas para o formato esperado (similar ao TeamGameLog)
                        # TeamGameLog usa: Team_ID, Game_ID, GAME_DATE, MATCHUP, WL, etc.
                        # LeagueGameLog usa: TEAM_ID, GAME_ID, GAME_DATE, MATCHUP, WL, etc.
                        # A maioria das colunas já está no formato correto, só precisamos renomear algumas
                        column_mapping = {
                            'TEAM_ID': 'Team_ID',
                            'TEAM_ABBREVIATION': 'Team_Abbreviation',
                            'TEAM_NAME': 'Team_Name'
                        }
                        df = df.rename(columns=column_mapping)
                        
                        # Garantir que as colunas numéricas sejam numéricas
                        numeric_columns = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT',
                                           'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                                           'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'MIN']
                        
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        print(f"INFO - Dados obtidos via LeagueGameLog: {len(df)} jogos")
                        return df
                    else:
                        print(f"WARNING - LeagueGameLog retornou dados, mas nenhum jogo encontrado para team_id {team_id}")
                        return pd.DataFrame()
                else:
                    print(f"WARNING - LeagueGameLog não tem coluna TEAM_ID")
                    return pd.DataFrame()
            else:
                print(f"WARNING - LeagueGameLog também retornou vazio")
                return pd.DataFrame()
            
        except Exception as e:
            print(f"Erro ao obter log de jogos do time: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_league_team_stats(self):
        """Obtém estatísticas agregadas de todos os times"""
        try:
            time.sleep(0.6)  # Rate limiting
            league_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=self.season,
                season_type_all_star=self.season_type
            )
            
            df = league_stats.league_dash_team_stats.get_data_frame()
            
            if df is None or df.empty:
                return None
            
            # Garantir que as colunas numéricas sejam numéricas
            numeric_columns = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT',
                               'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                               'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Erro ao obter estatísticas da liga: {e}")
            return None
    
    def extract_opponent_team_id(self, matchup_str):
        """Extrai o ID do time oponente do campo MATCHUP"""
        try:
            # MATCHUP geralmente tem formato: "LAL @ GSW" ou "GSW vs. LAL"
            if '@' in matchup_str:
                # Jogo fora de casa
                parts = matchup_str.split('@')
                if len(parts) == 2:
                    opponent_abbr = parts[1].strip()
                    team = teams.find_team_by_abbreviation(opponent_abbr)
                    if team:
                        return team['id']
            elif 'vs.' in matchup_str or 'VS' in matchup_str:
                # Jogo em casa
                parts = matchup_str.replace('vs.', 'VS').split('VS')
                if len(parts) == 2:
                    opponent_abbr = parts[1].strip()
                    team = teams.find_team_by_abbreviation(opponent_abbr)
                    if team:
                        return team['id']
            return None
        except Exception as e:
            print(f"Erro ao extrair ID do oponente: {e}")
            return None
    
    def is_home_game(self, matchup_str):
        """Determina se é jogo em casa baseado no MATCHUP"""
        return 'vs.' in matchup_str or 'VS' in matchup_str

