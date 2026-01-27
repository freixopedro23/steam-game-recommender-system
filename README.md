# 🎮 Game Matcher AI - Steam Recommendation System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

Um sistema de recomendação de jogos inteligente que utiliza **Processamento de Linguagem Natural (NLP)** e **Filtragem Baseada em Conteúdo** para sugerir novos jogos com base nos títulos que o usuário já gosta.

O projeto utiliza dados reais da Steam, processando descrições, tags, gêneros e desenvolvedores para calcular a similaridade matemática entre mais de 30.000 jogos.

---

## 📸 Screenshots

![App Screenshot](<img width="1914" height="912" alt="image" src="https://github.com/user-attachments/assets/56e18df4-97c4-4588-8998-8d6aae1aa80f" />)

---

## ✨ Funcionalidades

* **Busca Inteligente:** Encontre qualquer jogo da base de dados Steam.
* **Recomendação por Similaridade:** Algoritmo *Cosine Similarity* treinado em uma "sopa" de metadados (Tags + Gênero + Dev).
* **Filtros Dinâmicos:**
    * 🚸 **Controle Parental:** Filtre por classificação etária.
    * ⭐ **Qualidade:** Defina uma % mínima de aprovação da comunidade.
    * 💻 **Plataforma:** Filtre jogos compatíveis com Windows, Mac ou Linux.
* **Buffer de Candidatos:** O sistema analisa os Top 50 similares antes de aplicar os filtros, garantindo que você sempre receba 5 recomendações válidas.

---

## 🛠️ Arquitetura do Projeto

O projeto segue um pipeline de Engenharia de Machine Learning robusto:

1.  **ETL (`etl_steam.py`):**
    * Ingestão de dados brutos (`csv`).
    * Limpeza de strings (Regex) e tratamento de nulos.
    * Carga em Banco de Dados SQL (`sqlite`).
2.  **Modelagem (`model_training.py`):**
    * **Feature Engineering:** Criação de uma *Bag of Words* ponderada (Tags têm peso maior).
    * **Vetorização:** Uso de `CountVectorizer` (Scikit-Learn).
    * **Cálculo:** Matriz de Similaridade de Cossenos.
    * **Persistência:** Salvamento do modelo em arquivos `.pkl`.
3.  **App (`app.py`):**
    * Interface Front-end construída com **Streamlit**.
    * Carregamento otimizado de modelos com Cache.

---

## 🚀 Como Rodar Localmente

Siga os passos abaixo para testar em sua máquina:

### 1. Clone o repositório
```bash
git clone [https://github.com/SEU_USUARIO/game-matcher-ai.git](https://github.com/SEU_USUARIO/game-matcher-ai.git)
cd game-matcher-ai

---

### 2. Instale as dependências
```bash
pip install -r requirements.txt

---

### 3. Pipeline de Dados (Full Refresh)
Para criar o banco de dados e treinar o modelo do zero, execute o pipeline principal com a flag de reset:
```bash
python main.py --reset

Isso irá executar o ETL, criar o banco SQLite e gerar os arquivos .pkl na pasta models/.

---

### 4. Inicie o App
```bash
streamlit run app.py
