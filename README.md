# 🎮 Game Matcher AI — Steam Game Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Content--Based-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Visão Geral

**Game Matcher AI** é um sistema de recomendação de jogos baseado em **Machine Learning e NLP**, desenvolvido para sugerir novos títulos da Steam a partir dos jogos que o usuário já aprecia.

O projeto utiliza **Filtragem Baseada em Conteúdo**, processando **tags, gêneros, desenvolvedores e publishers** para calcular similaridade semântica entre **mais de 30.000 jogos reais da Steam**.

O foco não é apenas o modelo, mas sim **todo o pipeline de dados**, desde a ingestão até a entrega do produto final via aplicação web interativa.

---

## 🎯 Problema de Negócio

Encontrar novos jogos relevantes em plataformas com milhares de opções é um desafio comum para usuários da Steam.

Este projeto busca responder:

> *“Quais jogos têm maior chance de agradar um jogador, considerando apenas suas preferências explícitas e os metadados dos jogos?”*

---

## 🧠 Abordagem Técnica

O sistema utiliza uma arquitetura **end-to-end**, composta por:

- ETL estruturado
- Banco de dados relacional
- Feature Engineering com NLP
- Modelo de Similaridade Vetorial
- Interface web para consumo final

---

## 📸 Demonstração

<img src="https://github.com/user-attachments/assets/56e18df4-97c4-4588-8998-8d6aae1aa80f" alt="App Screenshot" width="100%">

---

## ✨ Funcionalidades

- 🔍 **Busca Inteligente** por qualquer jogo presente na base
- 🎯 **Recomendação por Similaridade Semântica**
  - Baseada em *Cosine Similarity*
  - Vetorização de metadados textuais
- 🎛️ **Filtros Dinâmicos**
  - 🚸 Classificação etária
  - ⭐ Percentual mínimo de aprovação da comunidade
  - 💻 Compatibilidade com Windows, Mac e Linux
- 🧠 **Buffer Inteligente de Candidatos**
  - O modelo analisa os **Top 50 jogos mais similares**
  - Após isso, aplica filtros para garantir **5 recomendações válidas**

---

## 🏗️ Arquitetura do Projeto

O projeto segue boas práticas de **Engenharia de Machine Learning**, organizado em camadas:

### 1️⃣ ETL — `etl_steam.py`
- Ingestão do dataset bruto (Kaggle)
- Limpeza de dados com Regex
- Tratamento de valores nulos
- Persistência em banco **SQLite**

### 2️⃣ Modelagem — `model_training.py`
- **Feature Engineering**
  - Criação de uma *feature soup* (Tags + Gêneros + Dev + Publisher)
  - Peso maior para *tags* (maior relevância semântica)
- **Vetorização**
  - `CountVectorizer` (Scikit-learn)
- **Modelo**
  - Similaridade de Cossenos
- **Persistência**
  - Modelos salvos em arquivos `.pkl`

### 3️⃣ Aplicação — `app.py`
- Interface web desenvolvida com **Streamlit**
- Cache de modelos para melhor performance
- Filtros interativos em tempo real

---

## 🚀 Como Executar Localmente

### 1️⃣ Clonar o repositório
git clone https://github.com/freixopedro23/steam-game-recommender-system.git
cd steam-game-recommender-system

### 2️⃣ Criar ambiente virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

### 3️⃣ Instalar dependências
pip install -r requirements.txt

### 4️⃣ Executar pipeline completo
python main.py --reset

### 5️⃣ Rodar aplicação
streamlit run app.py

---

## 📊 Dataset

- Fonte: Kaggle — Steam Games Dataset
- Registros: ~30.000 jogos
- Campos principais:
   - Tags
   - Gêneros
   - Desenvolvedor
   - Publisher
   - Avaliações da comunidade
   - Compatibilidade por OS

---

## ⚠️ Limitações Atuais

- Modelo 100% content-based (não utiliza histórico de usuários)
- Similaridade calculada sobre metadados (não considera gameplay real)
- Matriz de similaridade pode ser custosa em memória para grandes volumes

---

## 🔮 Próximos Passos (Evoluções Planejadas)

- 🔄 Migrar de CountVectorizer para TF-IDF
- ⚡ Implementar Top-K Nearest Neighbors (evitar matriz NxN)
- 🧩 Versão híbrida (content-based + popularidade)
- 👥 Integração com dados de reviews/playtime (Collaborative Filtering)
- 📦 Dockerização do projeto
- 🤖 Deploy em cloud (Streamlit Cloud / Hugging Face Spaces)

---

## 👨‍💻 Autor

**Pedro Freixo**
🎓 Data Science — FIAP
🎓 Profissão: Analista de Dados — EBAC

🔗 GitHub: https://github.com/freixopedro23
🔗 LinkedIn: https://www.linkedin.com/in/pedro-freixo-71b7ab212/
