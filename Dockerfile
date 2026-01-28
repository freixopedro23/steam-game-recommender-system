# Usa uma imagem leve do Python 3.10
FROM python:3.10-slim

# Define a pasta de trabalho dentro do container
WORKDIR /app

# Instala dependências do sistema operacional necessárias
# build-essential e git são vitais para algumas libs python
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copia os arquivos do projeto para dentro do container
COPY . .

# Instala as dependências do Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Expõe a porta padrão do Streamlit
EXPOSE 8501

# Checagem de saúde (Healthcheck) para garantir que o app está vivo
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Comando de inicialização:
# 1. Tenta deletar o DB antigo (se houver volume persistente)
# 2. Roda o Pipeline completo (cria DB, ETL, Treina)
# 3. Sobe o App
CMD python main.py --reset && streamlit run app.py --server.port=8501 --server.address=0.0.0.0