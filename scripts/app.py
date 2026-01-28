import os
import pandas as pd
import pickle
import streamlit as st
import time
from model_training import train_model
from db_setup import init_db
from etl_steam import run_etl

st.set_page_config(
    page_title="Game Matcher Steam",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        height: 3em;
        border-radius: 10px;
    }
    div.stImage > img {
        border-radius: 10px;
        transition: transform .2s;
    }
    div.stImage > img:hover {
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    path_models = 'models'

    # Verifica se existem
    required_files = ['neighbors_indices.pkl', 'neighbors_distances.pkl', 'dataframe.pkl', 'indices_map.pkl']
    if not all([os.path.exists(os.path.join(path_models, f)) for f in required_files]):
        return None, None, None, None

    try:
        with open(os.path.join(path_models, 'neighbors_indices.pkl'), 'rb') as f:
            knn_indices = pickle.load(f)
        with open(os.path.join(path_models, 'neighbors_distances.pkl'), 'rb') as f:
            knn_distances = pickle.load(f)
        with open(os.path.join(path_models, 'dataframe.pkl'), 'rb') as f:
            df = pickle.load(f)
        with open(os.path.join(path_models, 'indices_map.pkl'), 'rb') as f:
            indices_map = pickle.load(f)
        return knn_indices, knn_distances, df, indices_map
    except Exception:
        return None, None, None, None

# Tenta carregar
knn_indices, knn_distances, df_games, indices_map = load_data()


with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/c1/Steam_Logo.png/640px-Steam_Logo.png", width=150)
    st.title("Filtros de Busca")
    st.markdown("Refine sua recomendação:")

    # Filtro de idade
    st.markdown("---")
    st.markdown("🚸 **Controle Parental**")
    age_filter = st.slider(
        "Classificação etária do jogo:",
        min_value=0,
        max_value=18,
        value=18,
        step=1,
        help="Exclui jogos com classificação etária acima deste valor."
    )

    # Filtro de qualidade
    st.markdown("---")
    st.markdown("⭐ **Qualidade Mínima**")
    score_filter = st.slider(
        "% de Avaliações Positivas:",
        min_value=0,
        max_value=100,
        value=50,
        step=1,
        help="Mostra apenas jogos com aprovação da comunidade acima de X%"
    )

    # Filtro de sistema operacional
    st.markdown("---")
    st.markdown("💻 **Plataforma**")
    os_windows = st.checkbox("Windows", value=True)
    os_mac = st.checkbox("Mac", value=True)
    os_linux = st.checkbox("Linux", value=True)

    st.info("💡 **Dica:** Digite o nome do seu jogo favorito na caixa de busca para encontrar títulos similares baseados em gênero, tags e desenvolvedores.")

if df_games is None:
    st.warning("⚠️ Os modelos de IA ainda não foram gerados.")
    st.info("Como é a primeira execução, precisamos processar o banco de dados. Isso pode levar cerca de 1 minuto.")

    if st.button("🚀 Iniciar Instalação Completa"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            # PASSO 1: Criar o Banco de Dados (Tabelas vazias)
            status_text.text("1/3: Criando Banco de Dados SQL...")
            init_db()
            progress_bar.progress(20)

            # PASSO 2: Rodar ETL (CSV -> SQL)
            status_text.text("2/3: Carregando dados do CSV para o SQL (ETL)...")
            # O ETL precisa que o arquivo CSV esteja na pasta 'data/' no GitHub!
            with st.spinner('Importando dados brutos...'):
                run_etl()
            progress_bar.progress(60)

            # PASSO 3: Treinar Modelo (SQL -> PKL)
            status_text.text("3/3: Treinando Inteligência Artificial...")
            with st.spinner('Processando NLP e Matriz de Vizinhos...'):
                train_model()
            progress_bar.progress(90)

            # FINALIZAÇÃO
            status_text.text("✅ Tudo pronto! Reiniciando...")
            progress_bar.progress(100)

            # Limpa o cache para carregar os novos arquivos
            load_data.clear()
            time.sleep(1)
            st.rerun()

        except Exception as e:
            st.error(f"❌ Erro Crítico no Pipeline: {e}")
            st.error("Verifique se a pasta 'data/' e o arquivo CSV foram enviados para o GitHub.")

    st.stop()

st.title("Game Matcher Steam")
st.subheader("Decida sua próxima gameplay baseado no que acabou de jogar.")

# Selecionando o jogo
game_list = sorted(df_games['nm_game'].unique())
game_option = st.selectbox(
    "Escolha um jogo que você curtiu:",
    game_list,
    index=None,
    placeholder="Digite para pesquisar (ex: Elden Ring)..."
)

if game_option:
    if st.button("🔍 Encontrar Recomendações"):
        try:
            # ID do jogo escolhido
            idx = indices_map[game_option]

            # Pega os 50 vizinhos
            neighbor_indices = knn_indices[idx][1:]
            neighbor_distances = knn_distances[idx][1:]

            recommended_games = []

            for i, game_idx in enumerate(neighbor_indices):
                game_data = df_games.iloc[game_idx]

                similarity_score = 1 - neighbor_distances[i]

                # Filtragem
                # 1. Idade
                game_age = game_data['vl_required_age'] if pd.notna(game_data['vl_required_age']) else 0
                if game_age > age_filter:
                    continue

                # 2. Aprovação
                game_score = game_data['vl_positive_ratio'] if pd.notna(game_data['vl_positive_ratio']) else 0
                if game_score < score_filter:
                    continue

                # 3. Sistema Operacional
                compatible = False
                if os_windows and game_data['bl_windows']: compatible = True
                if os_linux and game_data['bl_linux']: compatible = True
                if os_mac and game_data['bl_mac']: compatible = True

                if not compatible:
                    continue

                # Se passou em tudo, adiciona na lista final
                game_data['match_score'] = similarity_score
                recommended_games.append(game_data)

                if len(recommended_games) >= 5:
                    break

            if len(recommended_games) == 0:
                st.warning("😔 Nenhum jogo encontrado com esses filtros. Tente diminuir a nota mínima ou aumentar a idade.")
            else:
                st.divider()
                st.markdown(f"### Se você gosta de **{game_option}**, experimente:")

                cols = st.columns(len(recommended_games))

                for col, game_data in zip(cols, recommended_games):
                    with col:
                        # Imagem do jogo
                        img_url = game_data['ds_url_header']
                        # Verificando se é válida
                        if pd.isna(img_url) or img_url == '':
                            img_url = "https://via.placeholder.com/300x150?text=No+Image"

                        st.image(img_url, use_container_width=True)

                        # Título
                        st.markdown(f"**{game_data['nm_game']}**")

                        # Dados extras
                        genre = str(game_data['ds_genres']).split(',')[0] if game_data['ds_genres'] else "Game"
                        st.caption(f"🏷️ {genre}")

                        # Aprovação
                        st.caption(f"👍 Aprovação: {int(game_data['vl_positive_ratio'])}%")

                        score = game_data['vl_metacritic_score']
                        if score and score > 0:
                            st.caption(f"⭐ Metacritic: {score}")

                        # Ícones de OS
                        os_icons = ""
                        if game_data['bl_windows']: os_icons += "🪟 "
                        if game_data['bl_mac']: os_icons += "🍎 "
                        if game_data['bl_linux']: os_icons += "🐧 "
                        st.caption(os_icons)

                        if game_data['vl_required_age'] > 0:
                            st.caption(f"🔞 +{int(game_data['vl_required_age'])}")
                        else:
                            st.caption("✅ Livre")

        except Exception as e:
            st.error(f"Ops! Ocorreu um erro ao processar: {e}")