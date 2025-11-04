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
            
            team_log = teamgamelog.TeamGameLog(
                team_id=team_id,
                season=self.season,
                season_type_all_star=self.season_type
            )
            
            # Verificar resposta bruta primeiro
            raw_dict = team_log.get_dict()
            
            # Debug: verificar estrutura da resposta
            print(f"DEBUG - Tipo de raw_dict: {type(raw_dict)}")
            if raw_dict:
                print(f"DEBUG - Keys em raw_dict: {list(raw_dict.keys())}")
            
            # Verificar se há dados na resposta bruta
            has_data = False
            row_count = 0
            headers = None
            row_set = None
            
            # Verificar resultSets primeiro
            if raw_dict and 'resultSets' in raw_dict:
                result_sets = raw_dict['resultSets']
                print(f"DEBUG - resultSets encontrado: {type(result_sets)}, tamanho: {len(result_sets) if isinstance(result_sets, list) else 'N/A'}")
                if isinstance(result_sets, list) and len(result_sets) > 0:
                    first_set = result_sets[0]
                    print(f"DEBUG - Tipo do primeiro set: {type(first_set)}")
                    if isinstance(first_set, dict):
                        print(f"DEBUG - Keys no primeiro set: {list(first_set.keys())}")
                        if 'rowSet' in first_set:
                            row_set = first_set['rowSet']
                            row_count = len(row_set) if row_set else 0
                            print(f"DEBUG - rowSet encontrado: {row_count} linhas")
                            if row_count > 0:
                                has_data = True
                                print(f"DEBUG - Primeira linha: {row_set[0]}")
                        if 'headers' in first_set:
                            headers = first_set['headers']
                            print(f"DEBUG - Headers encontrados: {len(headers) if headers else 0} colunas")
                        if 'name' in first_set:
                            print(f"DEBUG - Nome do dataset: {first_set['name']}")
            
            # Tentar obter DataFrame usando o método normal
            df = None
            try:
                # Verificar se o atributo team_game_log existe e foi carregado
                if hasattr(team_log, 'team_game_log'):
                    if team_log.team_game_log is not None:
                        # Verificar se o data_set tem dados
                        data_dict = team_log.team_game_log.get_dict()
                        print(f"DEBUG - team_game_log data_dict: {list(data_dict.keys()) if isinstance(data_dict, dict) else type(data_dict)}")
                        if isinstance(data_dict, dict) and 'headers' in data_dict:
                            print(f"DEBUG - Headers no data_dict: {len(data_dict['headers']) if data_dict['headers'] else 0}")
                            print(f"DEBUG - Data no data_dict: {len(data_dict.get('data', []))}")
                        
                        df = team_log.team_game_log.get_data_frame()
                        print(f"DEBUG - DataFrame obtido via team_game_log: {df is not None}, shape: {df.shape if df is not None else 'N/A'}")
                    else:
                        print(f"DEBUG - team_game_log existe mas é None")
                else:
                    print(f"DEBUG - team_game_log não existe no objeto")
                    # Tentar usar get_data_frames() como alternativa
                    try:
                        data_frames = team_log.get_data_frames()
                        print(f"DEBUG - get_data_frames() retornou: {len(data_frames) if data_frames else 0} dataframes")
                        if data_frames and len(data_frames) > 0:
                            df = data_frames[0]
                            print(f"DEBUG - DataFrame obtido via get_data_frames(): shape: {df.shape if df is not None else 'N/A'}")
                    except Exception as e2:
                        print(f"DEBUG - Erro ao usar get_data_frames(): {e2}")
            except Exception as e:
                print(f"WARNING - Erro ao obter DataFrame via team_game_log: {e}")
                import traceback
                traceback.print_exc()
            
            # Se DataFrame é None ou vazio, mas há dados na resposta bruta, construir manualmente
            if (df is None or (df.empty if df is not None else True)) and has_data and headers and row_set:
                try:
                    df = pd.DataFrame(row_set, columns=headers)
                    print(f"INFO - DataFrame construído manualmente: {len(df)} linhas, {len(df.columns)} colunas")
                except Exception as e:
                    print(f"WARNING - Erro ao construir DataFrame manualmente: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Verificar se retornou algo válido
            if df is None:
                return None
            
            # Se retornou um DataFrame vazio
            if df.empty:
                return pd.DataFrame()
            
            # Garantir que as colunas numéricas sejam numéricas
            numeric_columns = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT',
                               'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                               'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'MIN']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
            
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

