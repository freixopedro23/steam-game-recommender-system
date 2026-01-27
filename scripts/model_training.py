import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

    # Carregar dados (Limitando aos Top 35k mais avaliados para performance)
    query = """
            SELECT * \
            FROM game
            ORDER BY vl_recommendations DESC LIMIT 35000 \
            """
    print("📦 Carregando os jogos do SQL...")
    df_treino = pd.read_sql(query, engine)

    print(f"Antes da limpeza: {len(df_treino)} linhas")
    df_treino = df_treino.drop_duplicates(subset='nm_game', keep='first')
    df_treino = df_treino.reset_index(drop=True)
    print(f"Linhas carregadas do Banco: {len(df_treino)} linhas")

    # "Feature Soup"
    # Juntamos Gêneros, Tags, Desenvolvedores e Descrição numa única string
    print("🍲 Cozinhando a 'Sopa de Features' (NLP)...")
    def create_soup(x):
        # Tratamento para garantir que tudo seja string
        genres = str(x['ds_genres']) if x['ds_genres'] else ''
        tags = str(x['ds_tags']) if x['ds_tags'] else ''
        developers = str(x['ds_developer']) if x['ds_developer'] else ''
        publisher = str(x['ds_publisher']) if x['ds_publisher'] else ''
        # Damos peso duplicado para TAGS, pois elas definem melhor o jogo
        return genres + ' ' + tags + ' ' + tags + ' ' + developers + ' ' + publisher

    df_treino['soup'] = df_treino.apply(create_soup, axis=1)

    # NLP - stop-words: necessário para tirar o the, and
    print("🧮 Vetorizando textos (CountVectorizer)...")
    count = CountVectorizer(stop_words='english', min_df=5)
    count_matrix = count.fit_transform(df_treino['soup'])
    # Cálculo de Similaridade (A Mágica)
    print("📐 Calculando similaridade de Cossenos...")
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    indices = pd.Series(df_treino.index, index=df_treino['nm_game'])

    # Salvar
    print("💾 Salvando arquivos .pkl...")

    # Criamos a pasta 'models' se não existir
    if not os.path.exists('models'):
        os.makedirs('models')

    # Salvamos a Matriz de Similaridade (O Cérebro)
    with open('models/similarity_matrix.pkl', 'wb') as f:
        pickle.dump(cosine_sim, f)
    # Salvamos o DataFrame (A Memória - para mostrar imagem e preço)
    with open('models/dataframe.pkl', 'wb') as f:
        pickle.dump(df_treino, f)
    # Salvamos o índice (O Mapa)
    with open('models/indices.pkl', 'wb') as f:
        pickle.dump(indices, f)

    print("✅ Modelo treinado e salvo na pasta 'models/'!")

if __name__ == "__main__":
    train_model()