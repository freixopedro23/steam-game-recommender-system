import os
import shutil
import argparse
from db_setup import init_db, DB_PATH
from etl_steam import run_etl
from model_training import train_model

# Definição dos caminhos
MODELS_DIR = 'models'

def clean_environment():
    """
    Função para deletar o Banco de Dados e os Modelos Salvos.
    Isso garante um reinício limpo (Cold Start).
    """
    print("\n🧹 Iniciando limpeza do ambiente...")

    # Deletar Banco de Dados
    if os.path.exists(DB_PATH):
        try:
            os.remove(DB_PATH)
            print(f"✅ Banco de dados deletado: {DB_PATH}")
        except PermissionError:
            print(f"❌ ERRO: Não foi possível deletar {DB_PATH}.")
            print("   -> Verifique se o Streamlit ou o DB Browser estão abertos e feche-os.")
            return False
    else:
        print("ℹ️  Nenhum banco de dados encontrado para deletar.")

    # Deletar Modelos (.pkl)
    if os.path.exists(MODELS_DIR):
        try:
            # Opção A: Deletar a pasta inteira e recriar
            shutil.rmtree(MODELS_DIR)
            os.makedirs(MODELS_DIR)
            print(f"✅ Pasta '{MODELS_DIR}' limpa e recriada.")
        except Exception as e:
            print(f"❌ Erro ao limpar pasta models: {e}")
    else:
        print("ℹ️  Pasta 'models' não existia.")

    return True

if __name__ == '__main__':
    # Configuração do Argument Parser (Leitor de comandos do terminal)
    parser = argparse.ArgumentParser(description="Pipeline do Game Recommender System")

    # Criamos a flag --reset. Se o usuário usar, a variável 'reset' vira True.
    parser.add_argument('--reset', action='store_true', help="Deleta DB e Modelos antigos antes de rodar.")

    args = parser.parse_args()

    # Lógica de Execução
    if args.reset:
        sucesso = clean_environment()
        if not sucesso:
            print("⚠️ Abortando pipeline devido a erro na limpeza.")
            exit()

    # Pipeline
    print("\n🚀 Iniciando Pipeline...")
    init_db()
    run_etl()
    train_model()
    print("\n🎉 Pipeline finalizado com sucesso!")