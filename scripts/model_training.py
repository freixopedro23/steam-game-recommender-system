import pandas as pd
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

# 1. Conectar no Banco
DATABASE_URL = 'sqlite:///game_recommender_system.db'
engine = create_engine(DATABASE_URL)


def train_model():
    print("🧠 Iniciando treinamento do modelo Content-Based...")

    # Carregar dados (Limitando aos Top 20k mais avaliados para performance)
    query = """
            SELECT * \
            FROM game
            ORDER BY vl_recommendations DESC LIMIT 35000 \
            """
    print("📦 Carregando os Top 20.000 jogos do SQL...")
    df_treino = pd.read_sql(query, engine)

    # "Feature Soup"
    # Juntamos Gêneros, Tags, Desenvolvedores e Descrição numa única string
    print("🍲 Cozinhando a 'Sopa de Features'...")

    def create_soup(x):
        # Tratamento para garantir que tudo seja string
        genres = str(x['ds_genres']) if x['ds_genres'] else ''
        tags = str(x['ds_tags']) if x['ds_tags'] else ''
        developers = str(x['ds_developer']) if x['ds_developer'] else ''
        desc = str(x['ds_short_description']) if x['ds_short_description'] else ''

        # Damos peso duplicado para TAGS, pois elas definem melhor o jogo
        return genres + ' ' + tags + ' ' + tags + ' ' + developers + ' ' + desc

    df_treino['soup'] = df_treino.apply(create_soup, axis=1)

    # NLP
    # stop_words='english' remove palavras como "the", "a", "is"
    print("🧮 Vetorizando textos (CountVectorizer)...")
    count = CountVectorizer(stop_words='english', min_df=5)
    count_matrix = count.fit_transform(df_treino['soup'])

    # Cálculo de Similaridade (A Mágica)
    print("kNN Calculando similaridade de Cossenos (Isso pode levar 1 min)...")
    cosine_sim = cosine_similarity(count_matrix, count_matrix)

    # Resetar Index para facilitar busca
    # Criamos um mapa reverso: Nome do Jogo -> Índice na Matriz
    df = df_treino.reset_index(drop=True)
    indices = pd.Series(df.index, index=df['nm_game']).drop_duplicates()

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
        pickle.dump(df, f)

    # Salvamos o índice (O Mapa)
    with open('models/indices.pkl', 'wb') as f:
        pickle.dump(indices, f)

    print("✅ Modelo treinado e salvo na pasta 'models/'!")

    # --- TESTE RÁPIDO NO TERMINAL ---
    print("\n--- 🧪 TESTE DE RECOMENDAÇÃO ---")
    try:
        # Vamos tentar pegar um jogo famoso que provavelmente está no Top 20k
        test_game = "Street Fighter V"
        if test_game in indices:
            idx = indices[test_game]
            sim_scores = list(enumerate(cosine_sim[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:6]  # Top 5 (ignorando o 0 que é ele mesmo)

            print(f"Quem jogou '{test_game}' também vai gostar de:")
            for i in sim_scores:
                print(f"- {df.iloc[i[0]]['nm_game']}")
        else:
            print(f"O jogo de teste '{test_game}' não entrou no Top 20k. Tente outro.")
    except Exception as e:
        print(f"Erro no teste: {e}")


if __name__ == "__main__":
    train_model()