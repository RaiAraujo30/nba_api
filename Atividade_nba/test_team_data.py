"""Script de teste para verificar dados de times"""
import sys
import os

# Adicionar o diretório atual ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from data_collector import NBADataCollector

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
    test_team_data()

