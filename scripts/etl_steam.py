import pandas as pd
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DB_FOLDER = 'database'
DB_NAME = 'game_recommender_system.db'

if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)

DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# Criando conexão
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()

def run_etl():
    print("Starting ETL... ")

    # Lendo o arquivo
    csv_path = 'data/games_march2025_full.csv'
    print(f"Lendo o arquivo {csv_path}")
    df_game = pd.read_csv(csv_path, on_bad_lines='skip')

    print(f"Tamanho do DataFrame original: {len(df_game)}")

    # Começando a limpeza
    df_game.dropna(subset=['appid', 'name'], inplace=True)
    df_game.drop_duplicates(subset=['appid'], inplace=True)
    df_game.drop_duplicates(subset=['name', 'developers'], inplace=True)

    # Evitado erros na coluna de percentual positivo total
    df_game['pct_pos_total'].fillna(0, inplace=True)

    # Limpando as colunas que contém listas
    cols_to_clean = ['genres', 'tags', 'developers', 'categories', 'publishers', ]
    for col in cols_to_clean:
        if col in df_game.columns:
            df_game[col] = df_game[col].str.replace(r"[\[\]']", "", regex=True)

    mapa_colunas_bd = {
        'appid': 'cd_game', 'name': 'nm_game', 'release_date': 'dt_release', 'price': 'vl_price',
        'header_image': 'ds_url_header', 'website': 'ds_website', 'support_url': 'ds_support_url',
        'required_age': 'vl_required_age', 'metacritic_score': 'vl_metacritic_score', 'pct_pos_total': 'vl_positive_ratio',
        'recommendations': 'vl_recommendations', 'windows': 'bl_windows', 'mac': 'bl_mac', 'linux': 'bl_linux',
        'developers': 'ds_developer', 'publishers': 'ds_publisher', 'categories': 'ds_categories', 'genres': 'ds_genres',
        'tags': 'ds_tags', 'short_description': 'ds_short_description'
    }
    df_game.rename(columns=mapa_colunas_bd, inplace=True)

    colunas_bd = [
        'cd_game', 'nm_game', 'dt_release', 'vl_price', 'ds_url_header', 'ds_website', 'ds_support_url',
        'vl_required_age', 'vl_metacritic_score', 'vl_positive_ratio', 'vl_recommendations', 'bl_windows',
        'bl_mac', 'bl_linux', 'ds_developer', 'ds_publisher', 'ds_categories', 'ds_genres','ds_tags',
        'ds_short_description'
    ]

    cols_existentes = [c for c in colunas_bd if c in df_game.columns]
    df_final = df_game[cols_existentes]

    print(f"Linhas prontas para inserção: {len(df_final)}")

    print(f"Iniciando inserção no Banco de Dados...")

    try:
        df_final.to_sql('game', con=engine, if_exists='append', index=False, chunksize=10000)
        print("ETL Completo! Banco populado")
    except Exception as e:
        print(f"Erro ao salvar o banco: {e}")

if __name__ == "__main__":
    run_etl()