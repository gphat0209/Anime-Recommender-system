# from dotenv import load_dotenv
# import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from qdrant_client import QdrantClient

# from qdrant_client.models import Distance, VectorParams, PointStruct
# from qdrant_client.http.models import VectorParams, Distance, SparseVectorParams, SparseIndexParams
from qdrant_client.http.models import PointStruct, SparseVector

from collections import defaultdict
from qdrant_client.http import models
# from langchain_community.embeddings import GPT4AllEmbeddings
# import hashlib
# import json
# load_dotenv()

# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
#embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key = GOOGLE_API_KEY)
# COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
# embedding_model = GPT4AllEmbeddings()
# qdrant = QdrantClient(path="../database/qdarnt_anime_db")

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


def search_anime_hybrid(qdrant, collection_name, embedding_model, query_text):

    query_dense = embedding_model.embed_query(query_text)
    query_sparse = get_sparse_vector(query_text)  
    indices, values = query_sparse.indices, query_sparse.values 

    results = qdrant.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=models.SparseVector(indices=indices, values=values),
                using="sparse",
                limit=20,
            ),
            models.Prefetch(
                query=query_dense,   # list / array floats
                using="dense",
                limit=20,
            ),
        ],
        query=models.FusionQuery(
            fusion=models.Fusion.RRF,
            weights={
                "dense":2.0,
                "sparse":1.0
            }),
        limit=15
    )
    output = [
        {
            "Name": r.payload.get("name", "N/A"),
            "Synopsis": r.payload.get("synopsis", "N/A"),
            "Genres":  r.payload.get("genres", "").split(", ") if r.payload.get("genres") else [],
            "ID": r.payload.get("id", ""),
            "Type": r.payload.get("type", ""),
            "Score": r.payload.get("score", ""),
            "Episode": r.payload.get("episode", ""),
            "Aired": r.payload.get("aired", ""),
            "Matching_score": r.score
        }
        for r in results.points
    ]

    return output

def search_anime_semantic(qdrant, collection_name, embedding_model, query_text):

    query_dense = embedding_model.embed_query(query_text)
    

    results = qdrant.query_points(
    collection_name="anime",
    query=query_dense,   # chỉ cần đưa dense vector
    using="dense",       # tên vector
    limit=15
    )
    output = [
        {
            "Name": r.payload.get("name", "N/A"),
            "Synopsis": r.payload.get("synopsis", "N/A"),
            "Genres":  r.payload.get("genres", "").split(", ") if r.payload.get("genres") else [],
            "ID": r.payload.get("id", ""),
            "Type": r.payload.get("type", ""),
            "Score": r.payload.get("score", ""),
            "Episode": r.payload.get("episode", ""),
            "Aired": r.payload.get("aired", ""),
            "Matching_score": r.score
        }
        for r in results.points
    ]

    return output

# results = search_anime(qdrant, COLLECTION_NAME, embedding_model, "adventures with elves")

# print(results)