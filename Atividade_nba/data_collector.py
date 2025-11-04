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

    def get_gsw_roster(self):
        """Retorna lista de jogadores do Golden State Warriors"""
        try:
            from nba_api.stats.endpoints import commonteamroster

            gsw_id = 1610612744

            time.sleep(0.6)  # Rate limiting
            roster = commonteamroster.CommonTeamRoster(
                team_id=gsw_id, season=self.season
            )

            roster_df = roster.common_team_roster.get_data_frame()
            
            players_list = []
            for _, row in roster_df.iterrows():
                players_list.append({
                    'id': row['PLAYER_ID'],
                    'full_name': row['PLAYER']
                })
            
            players_list.sort(key=lambda x: x['full_name'])
            
            return players_list

        except Exception as e:
            print(f"Erro ao buscar roster do GSW: {e}")
            import traceback
            traceback.print_exc()
            return []

    def find_gsw_player(self, player_name):
        """Busca um jogador do Golden State Warriors pelo nome"""
        try:
            from nba_api.stats.endpoints import commonteamroster

            gsw_id = 1610612744

            time.sleep(0.6)  # Rate limiting
            roster = commonteamroster.CommonTeamRoster(
                team_id=gsw_id, season=self.season
            )

            roster_df = roster.common_team_roster.get_data_frame()

            player_list = players.find_players_by_full_name(player_name)

            if not player_list:
                return None

            player = player_list[0]
            player_id = player["id"]

            if player_id in roster_df["PLAYER_ID"].values:
                return player
            else:
                return None  # Jogador não está no GSW

        except Exception as e:
            print(f"Erro ao buscar jogador do GSW: {e}")
            import traceback

            traceback.print_exc()
            return None

    def find_team(self, team_name):
        """Busca um time pelo nome"""
        try:
            team_list = teams.find_teams_by_full_name(team_name)

            if not team_list:
                team_list = teams.find_teams_by_nickname(team_name)

            if not team_list:
                team_list = teams.find_teams_by_city(team_name)

            if not team_list:
                team_obj = teams.find_team_by_abbreviation(team_name.upper())
                if team_obj:
                    team_list = [team_obj]

            if not team_list:
                return None

            if len(team_list) == 1:
                return team_list[0]

            team_name_lower = team_name.lower()
            for team in team_list:
                if team_name_lower in team['full_name'].lower():
                    return team

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

            if df is None:
                return None

            if df.empty:
                return pd.DataFrame()

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

            df = None
            try:
                if hasattr(team_log, 'team_game_log') and team_log.team_game_log is not None:
                    df = team_log.team_game_log.get_data_frame()
            except Exception as e:
                print(f"WARNING - Erro ao obter DataFrame via team_game_log: {e}")

            if df is not None and not df.empty:
                numeric_columns = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT',
                                   'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                                   'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'MIN']

                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                return df

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
                if 'TEAM_ID' in df_all.columns:
                    df = df_all[df_all['TEAM_ID'] == team_id].copy()

                    if not df.empty:
                        column_mapping = {
                            'TEAM_ID': 'Team_ID',
                            'TEAM_ABBREVIATION': 'Team_Abbreviation',
                            'TEAM_NAME': 'Team_Name'
                        }
                        df = df.rename(columns=column_mapping)

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
            if '@' in matchup_str:
                parts = matchup_str.split('@')
                if len(parts) == 2:
                    opponent_abbr = parts[1].strip()
                    team = teams.find_team_by_abbreviation(opponent_abbr)
                    if team:
                        return team['id']
            elif 'vs.' in matchup_str or 'VS' in matchup_str:
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
