import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pickle
import os

DB_FOLDER = 'database'
DB_NAME = 'game_recommender_system.db'
if not os.path.exists(DB_FOLDER):
    os.makedirs(DB_FOLDER)
DB_PATH = os.path.join(DB_FOLDER, DB_NAME)

# Criando conexão
DATABASE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(DATABASE_URL)

def train_model():
    print("🧠 Iniciando treinamento...")

    # Carregar dados
    query = """
            SELECT * \
            FROM game
            WHERE vl_recommendations >= 20 \
            """
    print("📦 Carregando os jogos do SQL...")
    df_treino = pd.read_sql(query, engine)

    df_treino = df_treino.drop_duplicates(subset='nm_game', keep='first').reset_index(drop=True)
    print(f"Jogos para treino: {len(df_treino)}")

    # "Feature Soup"
    # Juntamos Gêneros, Tags, Desenvolvedores e Descrição numa única string
    print("🍲 Cozinhando a 'Sopa de Features' (NLP)...")
    def create_soup(x):
        # Tratamento para garantir que tudo seja string
        genres = str(x['ds_genres']) if pd.notna(x['ds_genres']) and x['ds_genres'] else ''
        tags = str(x['ds_tags']) if pd.notna(x['ds_tags']) and x['ds_tags'] else ''
        developers = str(x['ds_developer']) if pd.notna(x['ds_developer']) and x['ds_developer'] else ''
        publisher = str(x['ds_publisher']) if pd.notna(x['ds_publisher']) and x['ds_publisher'] else ''
        short_desc = str(x['ds_short_description']) if pd.notna(x['ds_short_description']) and x['ds_short_description'] else ''
        # Damos peso duplicado para TAGS, pois elas definem melhor o jogo
        return (
                genres + ' ' + genres + ' ' +
                tags + ' ' + tags + ' ' + tags + ' ' +
                developers + ' ' +
                publisher + ' ' +
                short_desc
        )


    df_treino['soup'] = df_treino.apply(create_soup, axis=1)

    # NLP - stop-words: necessário para tirar o the, and
    print("🧮 Vetorizando com TF-IDF..")
    tfidf = TfidfVectorizer(stop_words='english', min_df=5, ngram_range=(1, 2))

    tfidf_matrix = tfidf.fit_transform(df_treino['soup'])
    print(f"Matriz Gerada: {tfidf_matrix.shape[0]} jogos x {tfidf_matrix.shape[1]} palavras/termos")

    # Cálculo de Similaridade (A Mágica)
    print("📐 Calculando similaridade de Cossenos...")
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_jobs=-1)
    model_knn.fit(tfidf_matrix)

    distances, indices = model_knn.kneighbors(tfidf_matrix, n_neighbors=50)

    # Salvar
    print("💾 Salvando arquivos .pkl...")

    # Criamos a pasta 'models' se não existir
    if not os.path.exists('models'):
        os.makedirs('models')

    # Salvamos apenas a matriz de índices (N_jogos x 50)
    with open('models/neighbors_indices.pkl', 'wb') as f:
        pickle.dump(indices, f)
    # Se quiser mostrar a % de match, salve 'distances' também
    with open('models/neighbors_distances.pkl', 'wb') as f:
        pickle.dump(distances, f)
    # Salvamos o índice (O Mapa)
    with open('models/dataframe.pkl', 'wb') as f:
        pickle.dump(df_treino, f)
    # O mapa de nomes
    indices_map = pd.Series(df_treino.index, index=df_treino['nm_game'])
    with open('models/indices_map.pkl', 'wb') as f:
        pickle.dump(indices_map, f)


    print("✅ Modelo treinado e salvo na pasta 'models/'!")

if __name__ == "__main__":
    train_model()