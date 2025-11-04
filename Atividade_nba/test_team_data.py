"""Script de teste para verificar dados de times"""
import sys
import os

# Adicionar o diretório atual ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_collector import NBADataCollector
from nba_api.stats.endpoints import teamgamelog
from nba_api.stats.library.parameters import SeasonTypeAllStar

def test_team_data_direct():
    """Testa diretamente com a API"""
    print("=" * 60)
    print("Testando diretamente com a API")
    print("=" * 60)
    
    # Testar com Lakers ID
    team_id = 1610612747
    season = "2023-24"
    
    print(f"Team ID: {team_id}")
    print(f"Season: {season}")
    print()
    
    try:
        # Tentar diretamente com a API
        team_log = teamgamelog.TeamGameLog(
            team_id=team_id,
            season=season,
            season_type_all_star=SeasonTypeAllStar.regular
        )
        
        # Verificar resposta bruta
        raw_dict = team_log.get_dict()
        print(f"Resposta recebida: {raw_dict is not None}")
        
        if raw_dict and 'resultSets' in raw_dict:
            result_sets = raw_dict['resultSets']
            print(f"ResultSets: {len(result_sets) if isinstance(result_sets, list) else 'N/A'}")
            
            if isinstance(result_sets, list) and len(result_sets) > 0:
                first_set = result_sets[0]
                if isinstance(first_set, dict):
                    print(f"Keys no primeiro set: {list(first_set.keys())}")
                    if 'rowSet' in first_set:
                        row_set = first_set['rowSet']
                        print(f"RowSet tamanho: {len(row_set) if row_set else 0}")
                        if row_set and len(row_set) > 0:
                            print(f"Primeira linha: {row_set[0]}")
                    if 'headers' in first_set:
                        headers = first_set['headers']
                        print(f"Headers: {len(headers) if headers else 0}")
                        print(f"Headers: {headers}")
        
        # Tentar obter DataFrame
        df = team_log.team_game_log.get_data_frame()
        print(f"\nDataFrame shape: {df.shape}")
        print(f"DataFrame empty: {df.empty}")
        
        if not df.empty:
            print(f"\nPrimeiras 3 linhas:")
            print(df.head(3))
        else:
            print("\n⚠️ DataFrame vazio!")
            
            # Verificar se o problema é com o Team ID
            print("\nVerificando todos os times disponíveis...")
            from nba_api.stats.static import teams
            all_teams = teams.get_teams()
            print(f"Total de times: {len(all_teams)}")
            
            # Procurar Lakers nos times
            lakers = [t for t in all_teams if 'Lakers' in t['full_name']]
            if lakers:
                print(f"\nLakers encontrados:")
                for l in lakers:
                    print(f"  - {l['full_name']}: ID={l['id']}, Abbreviation={l['abbreviation']}")
                    
    except Exception as e:
        print(f"❌ Erro: {e}")
        import traceback
        traceback.print_exc()

def test_team_data():
    """Testa busca de dados de times"""
    # Testar com diferentes temporadas
    seasons = ['2024-25', '2023-24']
    
    collector = NBADataCollector()
    
    # Testar com Lakers
    print("=" * 60)
    print("Testando busca de dados do Los Angeles Lakers")
    print("=" * 60)
    
    team = collector.find_team("Los Angeles Lakers")
    if team:
        print(f"Time encontrado: {team['full_name']}")
        print(f"ID: {team['id']}")
        print(f"Abreviação: {team['abbreviation']}")
        print()
        
        for season in seasons:
            print(f"\n{'='*60}")
            print(f"Testando temporada: {season}")
            print(f"{'='*60}")
            
            # Criar novo collector com temporada específica
            test_collector = NBADataCollector(season=season)
            df = test_collector.get_team_game_log(team['id'])
            
            if df is not None:
                if len(df) > 0:
                    print(f"✅ Dados encontrados: {len(df)} jogos")
                    print(f"Colunas: {df.columns.tolist()}")
                    print(f"\nPrimeiras 3 linhas:")
                    print(df.head(3))
                else:
                    print(f"⚠️ DataFrame vazio (sem jogos para esta temporada)")
            else:
                print(f"❌ Erro ao obter dados (retornou None)")
    else:
        print("❌ Time não encontrado!")

if __name__ == "__main__":
    test_team_data_direct()
    print("\n" + "=" * 60)
    test_team_data()

