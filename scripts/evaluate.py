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
        if tag_clean and tag_clean != "nan": cleaned_tags.add(tag_clean)
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
    top_k = 5

    with open('benchmarks/test_games.txt', 'r', encoding='utf-8') as arquivo:
        test_games = [linha.strip() for linha in arquivo]

    print(f"{'JOGO DE ENTRADA':<25} | {'TOP-1 RECOMENDAÇÃO':<40} | {'COSINE@5':<10} | {'TAGS@5':<10}")
    print("-" * 100)

    cosine_at_k_scores = []
    overlap_at_k_scores = []

    for game in test_games:
        if game not in indices_map:
            print(f"{game:<25} | (não encontrado no dataset)")
            continue

        idx = indices_map[game]
        original_tags = df.iloc[idx].get("ds_tags", "")

        neighbor_indices_topk = knn_indices[idx][1:1 + top_k]
        neighbor_distances_topk = knn_distances[idx][1:1 + top_k]

        # Cosine@K (média da similaridade do Top-K)
        similarities_topk = 1 - neighbor_distances_topk
        cosine_at_k = float(np.mean(similarities_topk))

        # Top-K recomendados
        recommended_games = df.iloc[neighbor_indices_topk]
        top1 = recommended_games.iloc[0]

        # Tags@K (média do overlap de tags no Top-K)
        overlaps = []
        for _, rec in recommended_games.iterrows():
            overlaps.append(calculate_tag_overlap(original_tags, rec.get("ds_tags", "")))
        tags_at_k = float(np.mean(overlaps)) if overlaps else 0.0

        cosine_at_k_scores.append(cosine_at_k)
        overlap_at_k_scores.append(tags_at_k)

        print(f"{game:<25} | {str(top1['nm_game'])[:40]:<40} | {cosine_at_k:.1%}     | {tags_at_k:.1%}")

    if cosine_at_k_scores:
        avg_cosine = float(np.mean(cosine_at_k_scores))
        std_cosine = float(np.std(cosine_at_k_scores))
        avg_tags = float(np.mean(overlap_at_k_scores))
        std_tags = float(np.std(overlap_at_k_scores))

        print("-" * 100)
        print(f"✅ avg_cosine@{top_k}: {avg_cosine:.1%}   | 📉 std_cosine@{top_k}: {std_cosine:.1%}")
        print(f"✅ avg_tags@{top_k}:   {avg_tags:.1%}   | 📉 std_tags@{top_k}:   {std_tags:.1%}")


if __name__ == "__main__":
    run_evaluation()