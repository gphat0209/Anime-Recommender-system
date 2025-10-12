# 🚀 Anime Recommender System: A system that can tell whether your anime is Worth-watching 

## 📝 Overview
Anime Recommender System is an AI-powered system that can handle your interest in anime, by simply search animes that are close to your query or evaluate an anime from its synopsis and genres

---

## 🗂️ Project Structure

```
ANIME RECOMMENDATION SYSTEM/
│
├── Backend/
│   ├── database/
│   ├── retrieve_preprocessing/
│   ├── services/
│   ├── main.py                # for running backend application
│   └── requirements.txt
│
├── Frontend/
│   ├── maim.py
│   ├── Dockerfile
│   ├── requirements.txt

├── Fine-tuning/
│   ├── anime_model/
│   ├── training_data/
│   ├── finetune.py
│   ├── prepare_data.py
│
├── Images/                 # Documentation and workflow images
│
├── docker-compose.yml
└── README.md
```

---


### 1. Retrieve Data & Preprocessing

- **Data Source (Crawling):**
Anime metadata and user reviews were crawled public anime databases (e.g., MyAnimeList).
Only anime titles with ≥ 100,000 user ratings were retained to ensure reliability and data quality.

- **Preprocessing:**
Each anime entry was normalized and cleaned. The processed dataset includes:

- Title

- Synopsis

- Genres

- Score / Rating

- Label (Quality Tier): Categorized as Top-tier, Worthwhile, Watchable, or Terrible based on the normalized score distribution.

- **Storage:**
Cleaned data are stored in PostgreSQL for structured access and analysis.

### 2. Anime Search & Query

- **Goal:**
Search a right anime based on the user input.

- **Data Storage:**
Store them embedding of each anime's synopsis in Qdrant vector database, that help capture the semantic meaning of each one.

- **Result:**
When user enter a description, the website will return every anime that has the plot matches to the input description, its details and also the link to MyAnimeList page.


### 3. Model Training and Anime Evaluating
- **Goal:**
Predict the quality tier of an anime (Top-tier / Worthwhile / Watchable / Terrible) based on its synopsis and genres.

- **Training Process:**

- Embedding Model:

Use a pre-trained sentence embedding model (Gemini Embedding) to encode synopsis + genres into semantic vectors.
These embeddings capture narrative structure, emotional tone, and genre patterns.

- Classifier:

Train a Logistic Regression classifier (or MLP variant) on top of embeddings.
The classifier learns to separate anime into quality categories in the embedding space.

- Evaluation:

Split dataset into train/test (e.g., 70/30).
Achieved accuracy: ~0.60 on training, ~0.45 on test — a good baseline for limited data scenarios.

- Deployment:

Serve as a FastAPI endpoint /anime/evaluate.

When a user inputs a new synopsis and selected genres:
1. The text is embedded.
2. The trained model predicts the label.
3. An LLM layer (Gemini) generates a natural comment explaining the reasoning and tone-adjusted feedback for the user.

## 📚 References

### Key GitHub Repositories
- [LangChain](https://github.com/langchain-ai/langchain) 🧠
- [FastAPI](https://github.com/tiangolo/fastapi) ⚡
- [Qdrant Vector Database](https://github.com/qdrant/qdrant) 🗄️
- [PostgreSQL](https://github.com/postgres/postgres) 🛢️

### Related Research Papers & Articles
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) 📙
- [Semantic Search with Vector Databases](https://towardsdatascience.com/semantic-search-with-vector-databases-5c6b4c3d8e4b) 🔎

---

## ✨ Credits

Project initiated by [Truong Cong Gia Phat](https://github.com/gphat0209).

---

## 📄 License

This project is licensed under the MIT License.



