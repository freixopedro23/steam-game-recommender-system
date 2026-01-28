import pandas as pd
import zipfile
import os

# Configuração
INPUT_CSV = 'data/games_march2025_full.csv'
OUTPUT_CSV = 'data/games_filtered.csv'
OUTPUT_ZIP = 'data/games.zip'

# Essas são as únicas colunas O elt lê
COLS_TO_KEEP = [
    'appid', 'name', 'release_date', 'price', 'header_image',
    'website', 'support_url', 'required_age', 'metacritic_score',
    'pct_pos_total', 'recommendations', 'windows', 'mac', 'linux',
    'developers', 'publishers', 'categories', 'genres', 'tags',
    'short_description'
]


def minify_dataset():
    print(f"Lendo monstro: {INPUT_CSV}...")

    # Lê APENAS o necessário
    try:
        df = pd.read_csv(INPUT_CSV, usecols=lambda c: c in COLS_TO_KEEP, on_bad_lines='skip')
    except ValueError:
        # Fallback caso alguma coluna tenha nome ligeiramente diferente
        print("⚠️ Aviso: Alguma coluna não foi encontrada. Lendo tudo e filtrando depois...")
        df = pd.read_csv(INPUT_CSV, on_bad_lines='skip')
        existing_cols = [c for c in COLS_TO_KEEP if c in df.columns]
        df = df[existing_cols]

    print(f"Linhas Originais: {len(df)}")

    # REMOVER JOGOS IRRELEVANTES (Filtro de Qualidade)
    df_clean = df[df['recommendations'] >= 50].copy()

    print(f"Linhas após limpeza: {len(df_clean)}")

    # SALVAR CSV LIMPO
    print("Salvando CSV reduzido...")
    df_clean.to_csv(OUTPUT_CSV, index=False)

    # ZIPAR COM COMPRESSÃO MÁXIMA
    print("Comprimindo ao máximo...")
    with zipfile.ZipFile(OUTPUT_ZIP, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(OUTPUT_CSV, arcname='games.csv')

    os.remove(OUTPUT_CSV)

    size_mb = os.path.getsize(OUTPUT_ZIP) / (1024 * 1024)
    print(f"✅ Sucesso! Novo tamanho do arquivo: {size_mb:.2f} MB")

    if size_mb < 25:
        print("🟢 Pode subir pelo SITE ou TERMINAL.")
    elif size_mb < 100:
        print("🟡 Atenção: Suba usando 'git push' pelo TERMINAL (O site limita em 25MB).")
    else:
        print("🔴 Ainda está grande. Aumente o filtro de recommendations.")


if __name__ == "__main__":
    minify_dataset()