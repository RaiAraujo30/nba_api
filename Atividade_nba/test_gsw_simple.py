"""
Teste simples para verificar dados do Golden State Warriors
"""

import sys
import os

# Adicionar diretório atual
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Adicionar diretório pai para acessar nba_api
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from nba_api.stats.endpoints import teamgamelog
    from nba_api.stats.static import teams
    from nba_api.stats.library.parameters import SeasonTypeAllStar
    
    # Buscar Golden State Warriors
    print("=== Testando busca do time ===")
    team_list = teams.find_teams_by_full_name("Golden State Warriors")
    print(f"Times encontrados: {len(team_list) if team_list else 0}")
    
    if team_list:
        team = team_list[0]
        print(f"ID: {team['id']}")
        print(f"Nome: {team['full_name']}")
        print(f"Abreviação: {team['abbreviation']}")
        print()
        
        # Testar temporada 2024-25
        print("=== Testando temporada 2024-25 ===")
        team_log_2024 = teamgamelog.TeamGameLog(
            team_id=team['id'],
            season='2024-25',
            season_type_all_star=SeasonTypeAllStar.regular
        )
        
        raw_dict_2024 = team_log_2024.get_dict()
        print(f"Resposta bruta - Keys: {list(raw_dict_2024.keys()) if raw_dict_2024 else 'None'}")
        
        if 'resultSets' in raw_dict_2024:
            result_sets = raw_dict_2024['resultSets']
            print(f"ResultSets encontrado: {len(result_sets) if isinstance(result_sets, list) else 'N/A'}")
            if isinstance(result_sets, list) and len(result_sets) > 0:
                first_set = result_sets[0]
                print(f"Primeiro set - Keys: {list(first_set.keys()) if isinstance(first_set, dict) else 'N/A'}")
                if 'rowSet' in first_set:
                    row_count = len(first_set['rowSet'])
                    print(f"RowSet com {row_count} linhas")
                    if row_count > 0:
                        print(f"Primeiras 3 linhas:")
                        for i, row in enumerate(first_set['rowSet'][:3]):
                            print(f"  Linha {i+1}: {row}")
        
        df_2024 = team_log_2024.team_game_log.get_data_frame()
        print(f"DataFrame - É None: {df_2024 is None}")
        print(f"DataFrame - Está vazio: {df_2024.empty if df_2024 is not None else 'N/A'}")
        print(f"DataFrame - Shape: {df_2024.shape if df_2024 is not None else 'N/A'}")
        print()
        
        # Testar temporada 2023-24 para comparação
        print("=== Testando temporada 2023-24 (para comparação) ===")
        team_log_2023 = teamgamelog.TeamGameLog(
            team_id=team['id'],
            season='2023-24',
            season_type_all_star=SeasonTypeAllStar.regular
        )
        
        df_2023 = team_log_2023.team_game_log.get_data_frame()
        print(f"DataFrame - É None: {df_2023 is None}")
        print(f"DataFrame - Está vazio: {df_2023.empty if df_2023 is not None else 'N/A'}")
        print(f"DataFrame - Shape: {df_2023.shape if df_2023 is not None else 'N/A'}")
        if df_2023 is not None and not df_2023.empty:
            print(f"Total de jogos 2023-24: {len(df_2023)}")
            print(f"Primeiras linhas:")
            print(df_2023.head())
        
except Exception as e:
    print(f"Erro: {e}")
    import traceback
    traceback.print_exc()

