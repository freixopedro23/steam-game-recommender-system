# 🎮 Game Matcher AI — Steam Game Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Recommender%20System-green)
![Status](https://img.shields.io/badge/Status-Deployed-success)

## 🌐 Demo Online

👉 **Teste o app ao vivo:**  
https://steam-game-recommender-system-rvsg9fqukzey6sq2do2d9h.streamlit.app/

---

## 📌 Visão Geral

**Game Matcher AI** é um sistema de recomendação de jogos da Steam baseado em **Machine Learning e NLP**, desenvolvido para sugerir novos títulos a partir de um jogo de referência escolhido pelo usuário.

O projeto utiliza **Filtragem Baseada em Conteúdo**, processando **tags, gêneros, desenvolvedores, publishers e descrições** para calcular similaridade semântica entre **mais de 30.000 jogos reais da Steam**.

O foco do projeto é demonstrar um **pipeline completo de dados**, desde a ingestão e modelagem até o deploy de uma aplicação web interativa.

---

## 🎯 Problema de Negócio

Com milhares de jogos disponíveis na Steam, encontrar títulos relevantes pode ser difícil, especialmente sem histórico de usuário.

Este projeto busca responder à pergunta:

> *“Quais jogos têm maior probabilidade de agradar um jogador, considerando apenas os metadados e similaridade de conteúdo?”*

---

## 🧠 Abordagem Técnica

O sistema foi desenvolvido seguindo uma arquitetura **end-to-end**, composta por:

- ETL estruturado
- Banco de dados relacional
- Feature Engineering com NLP
- Modelo de Similaridade Vetorial (Top-K)
- Aplicação web para consumo final

---

## 📸 Demonstração

<img src="https://github.com/user-attachments/assets/56e18df4-97c4-4588-8998-8d6aae1aa80f" alt="App Screenshot" width="100%">

---

## ✨ Funcionalidades

- 🔍 **Busca Inteligente** por qualquer jogo da base Steam
- 🎯 **Recomendação por Similaridade Semântica**
  - TF-IDF + Similaridade de Cossenos
  - Recomendação Top-K com KNN
- 🎛️ **Filtros Dinâmicos**
  - 🚸 Classificação etária
  - ⭐ Percentual mínimo de aprovação da comunidade
  - 💻 Compatibilidade com Windows, Mac e Linux
- 🧠 **Buffer Inteligente de Candidatos**
  - O modelo analisa os Top 50 jogos mais similares
  - Após os filtros, retorna até 5 recomendações válidas

---

## 🏗️ Arquitetura do Projeto

O projeto segue boas práticas de **Engenharia de Machine Learning**, organizado em camadas:

### 1️⃣ ETL — `etl_steam.py`
- Ingestão do dataset bruto (Kaggle)
- Limpeza de dados (regex, deduplicação e tratamento de nulos)
- Persistência em banco **SQLite**

### 2️⃣ Modelagem — `model_training.py`
- **Feature Engineering**
  - Criação de uma *feature soup* combinando:
    - Tags (peso maior)
    - Gêneros
    - Desenvolvedores
    - Publisher
    - Short description
- **Vetorização**
  - `TF-IDF Vectorizer`
  - `ngram_range=(1,2)` para capturar conceitos compostos
- **Modelo**
  - `NearestNeighbors` com métrica de cosseno (Top-K)
  - Sem cálculo de matriz NxN (escalável)
- **Persistência**
  - Modelos salvos em arquivos `.pkl`

### 3️⃣ Aplicação — `app.py`
- Interface web desenvolvida com **Streamlit**
- Cache de recursos para melhor performance
- Consumo direto dos artefatos do modelo treinado

---

## 📊 Avaliação do Modelo

Para validação da qualidade das recomendações, foi criado o script `evaluate.py`, que compara:

- **COSINE (IA):** similaridade calculada pelo modelo (1 − cosine distance)
- **TAGS (REAL):** overlap de tags usando índice de Jaccard (validação explicável)

### 🔍 Resultado do teste

JOGO DE ENTRADA | RECOMENDAÇÃO | COSINE (IA) | TAGS (REAL)
Stardew Valley | Moonstone Island | 55.7% | 48.1%
Counter-Strike 2 | Team Fortress 2 | 52.8% | 37.9%
Baldur's Gate 3 | Divinity: Original Sin 2 | 42.6% | 29.0%
Euro Truck Simulator 2 | American Truck Simulator | 56.3% | 73.9%
ELDEN RING | DARK SOULS™ III | 59.5% | 60.0%

✅ Média Similaridade do Modelo (IA): 53.4%
✅ Média Overlap de Tags (Validação): 49.8%

---

## 🚀 Como Executar Localmente

### 1️⃣ Clonar o repositório
```bash
git clone https://github.com/freixopedro23/steam-game-recommender-system.git
cd steam-game-recommender-system
```

### 2️⃣ Criar ambiente virtual
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3️⃣ Instalar dependências
```bash
pip install -r requirements.txt
```

### 4️⃣ Executar pipeline completo
```bash
python main.py --reset
```

### 5️⃣ Rodar aplicação
```bash
streamlit run app.py
```

---

## 📦 Dataset

- Fonte: Kaggle — Steam Games Dataset
- Registros Total: ~90.000 jogos
- Principais campos:
  - Tags
  - Gêneros
  - Desenvolvedor / Publisher
  - Avaliações da comunidade
  - Compatibilidade por sistema operacional

---

## Limitações Atuais

- Sistema 100% content-based
- Não utiliza histórico real de usuários
- Similaridade baseada apenas em metadados

---

## 🔮 Próximos Passos
- 🔄 Versão híbrida (conteúdo + popularidade)
- 👥 Collaborative filtering com reviews/playtime
- 🧠 Explicabilidade das recomendações no app
- 📦 Dockerização
- 🔄 Versionamento de modelos

---

## 👨‍💻 Autor
**Pedro Freixo**
🎓 Data Science — FIAP
🎓 Profissão: Analista de Dados — EBAC

🔗 GitHub: https://github.com/freixopedro23
🔗 LinkedIn: https://www.linkedin.com/in/pedro-freixo-71b7ab212/
