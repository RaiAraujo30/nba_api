"""
Script de teste para verificar dados do Golden State Warriors 2024-25
"""

import sys
import os

# Adicionar diretório atual ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_collector import NBADataCollector
from nba_api.stats.static import teams

# Testar busca do time
print("=== Testando busca do Golden State Warriors ===")
team = teams.find_teams_by_full_name("Golden State Warriors")
print(f"Time encontrado: {team}")
if team:
    print(f"ID: {team[0]['id']}")
    print(f"Nome completo: {team[0]['full_name']}")
    print(f"Abreviação: {team[0]['abbreviation']}")
    print()

# Testar coleta de dados
print("=== Testando coleta de dados para temporada 2024-25 ===")
collector = NBADataCollector(season='2024-25')

if team:
    team_id = team[0]['id']
    print(f"Buscando dados para Team ID: {team_id}")
    print(f"Temporada: 2024-25")
    print(f"Season Type: {collector.season_type}")
    print()
    
    print("Carregando dados...")
    df = collector.get_team_game_log(team_id)
    
    if df is None:
        print("❌ Erro: Retornou None")
    elif df.empty:
        print("⚠️  Retornou DataFrame vazio (sem dados)")
        print(f"Colunas: {df.columns.tolist() if hasattr(df, 'columns') else 'N/A'}")
    else:
        print(f"✅ Dados carregados com sucesso!")
        print(f"Total de jogos: {len(df)}")
        print(f"Colunas: {df.columns.tolist()}")
        print(f"\nPrimeiras linhas:")
        print(df.head())
    
    print()
    print("=== Testando temporada 2023-24 para comparação ===")
    collector_23_24 = NBADataCollector(season='2023-24')
    df_23_24 = collector_23_24.get_team_game_log(team_id)
    
    if df_23_24 is None:
        print("❌ Erro: Retornou None")
    elif df_23_24.empty:
        print("⚠️  Retornou DataFrame vazio (sem dados)")
    else:
        print(f"✅ Dados carregados com sucesso!")
        print(f"Total de jogos: {len(df_23_24)}")
        print(f"\nPrimeiras linhas:")
        print(df_23_24.head())

