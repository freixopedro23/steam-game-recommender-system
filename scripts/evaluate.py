import pandas as pd
import pickle
import os
import numpy as np

MODELS_DIR = 'models'


def load_models():
    try:
        # Carregamos também o neighbors_distances.pkl agora
        with open(os.path.join(MODELS_DIR, 'neighbors_indices.pkl'), 'rb') as f:
            knn_indices = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'neighbors_distances.pkl'), 'rb') as f:
            knn_distances = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'dataframe.pkl'), 'rb') as f:
            df = pickle.load(f)
        with open(os.path.join(MODELS_DIR, 'indices_map.pkl'), 'rb') as f:
            indices_map = pickle.load(f)
        return knn_indices, knn_distances, df, indices_map
    except FileNotFoundError:
        print("❌ Modelos não encontrados. Rode 'python main.py --reset' primeiro.")
        exit()


def normalize_tags(tag_string):
    """Limpa tags sujas no formato '{Tag Name: 1234}'"""
    if pd.isna(tag_string) or tag_string == '':
        return set()
    tag_string = str(tag_string).replace('{', '').replace('}', '')
    cleaned_tags = set()
    items = tag_string.split(',')
    for item in items:
        if ':' in item:
            tag_name = item.split(':')[0]
        else:
            tag_name = item
        tag_clean = tag_name.strip().lower()
        if tag_clean: cleaned_tags.add(tag_clean)
    return cleaned_tags


def calculate_tag_overlap(tags1, tags2):
    set1 = normalize_tags(tags1)
    set2 = normalize_tags(tags2)
    if len(set1) == 0: return 0.0
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    if len(union) == 0: return 0.0
    return len(intersection) / len(union)


def run_evaluation():
    knn_indices, knn_distances, df, indices_map = load_models()

    test_games = [
        "Stardew Valley",
        "Counter-Strike 2",
        "Baldur's Gate 3",
        "Euro Truck Simulator 2",
        "ELDEN RING"
    ]

    # Cabeçalho ajustado para mostrar as duas métricas
    print(f"{'JOGO DE ENTRADA':<25} | {'RECOMENDAÇÃO':<25} | {'COSINE (IA)':<12} | {'TAGS (REAL)':<12}")
    print("-" * 85)

    tag_scores = []
    model_scores = []

    for game in test_games:
        if game not in indices_map:
            continue

        idx = indices_map[game]
        original_tags = df.iloc[idx]['ds_tags']

        # Pega o Top 1 Vizinho (índice 1)
        neighbor_idx = knn_indices[idx][1]
        neighbor_dist = knn_distances[idx][1]

        # Converte distância em similaridade (1 - distância)
        # Cosine Distance vai de 0 (idêntico) a 1 (oposto)
        similarity_model = 1 - neighbor_dist

        recommended_game = df.iloc[neighbor_idx]

        # Calcula Overlap de Tags
        tag_score = calculate_tag_overlap(original_tags, recommended_game['ds_tags'])

        tag_scores.append(tag_score)
        model_scores.append(similarity_model)

        print(f"{game:<25} | {recommended_game['nm_game']:<25} | {similarity_model:.1%}      | {tag_score:.1%}")

    if tag_scores:
        avg_tag = np.mean(tag_scores)
        avg_model = np.mean(model_scores)
        print("-" * 85)
        print(f"✅ Média Similaridade do Modelo (O que a IA acha):  {avg_model:.1%}")
        print(f"✅ Média Overlap de Tags (Validação Explicável):   {avg_tag:.1%}")


if __name__ == "__main__":
    run_evaluation()