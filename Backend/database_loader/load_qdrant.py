from dotenv import load_dotenv
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    VectorParams, Distance,
    SparseVectorParams, SparseIndexParams,
    PointStruct, SparseVector
)
from collections import defaultdict
from langchain_community.embeddings import GPT4AllEmbeddings
import hashlib
import json

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")

def load_qdrant_data():
    # embedding_model = GoogleGenerativeAIEmbeddings(
    #     model=EMBEDDING_MODEL_NAME,
    #     google_api_key=GOOGLE_API_KEY
    # )
    sub_embedding_model = GPT4AllEmbeddings()

    qdrant = QdrantClient(host="qdrant", port=6333)

    # CREATE COLLECTION IF NOT EXISTS
    if not qdrant.collection_exists(collection_name=COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(size=384, distance=Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(on_disk=True)
                )
            }
        )

    def make_id(text: str):
        return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16) % (10**8)

    def dedup_sparse(indices, values):
        agg = defaultdict(float)
        for i, v in zip(indices, values):
            agg[i] += v
        new_indices, new_values = zip(*agg.items())
        return list(new_indices), list(new_values)

    def get_sparse_vector(text: str, vocab_size: int = 1000) -> SparseVector:
        tokens = text.lower().split()
        indices = [abs(hash(t)) % vocab_size for t in tokens]
        values = [1.0] * len(tokens)
        indices, values = dedup_sparse(indices, values)
        return SparseVector(indices=indices, values=values)

    def load_anime(embedding_model, qdrant, collection_name, anime):
        name = anime.get("Name", "")
        synopsis = anime.get("Sypnosis", "")
        genres = anime.get("Genres", [])
        id = anime.get("ID", "")
        score = anime.get("Scores", "")
        type = anime.get("Type", "")
        episode = anime.get("Episodes", "")
        aired = anime.get("Aired", "")

        if not synopsis:
            return  

        synopsis_embedding = embedding_model.embed_query(synopsis)
        synopsis_bm25_vector = get_sparse_vector(synopsis)

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=make_id(name),
                    vector={
                        "dense": synopsis_embedding,
                        "sparse": synopsis_bm25_vector
                    },
                    payload={
                        "name": name,
                        "synopsis": synopsis,
                        "genres": genres,
                        "id": id,
                        "score": score,
                        "type": type,
                        "episode": episode,
                        "aired": aired
                    }
                )
            ]
        )

    print("Loading anime data into Qdrant...")
    with open("/app/output.json", "r", encoding="utf-8") as f:
        anime_data = json.load(f)

    for anime in anime_data:
        load_anime(sub_embedding_model, qdrant, COLLECTION_NAME, anime)

    print("✨ Qdrant loading completed!")


# Allow standalone execution
if __name__ == "__main__":
    load_qdrant_data()
