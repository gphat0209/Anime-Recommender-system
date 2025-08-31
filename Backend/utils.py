import pandas as pd
import json
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack import Document

def json_loadf(file):
    with open(file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return pd.DataFrame(data)

def load_document_store(file_path):
    document_store = FAISSDocumentStore(embedding_dim=384, faiss_index_factory_str="Flat")
    
    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",  # Nhanh, nhẹ
        use_gpu=True)
    with open(file_path, 'r', encoding='utf-8') as f:
        anime_data = json.load(f)

    docs = [
        Document(content=anime["sypnosis"], meta={"title": anime["title"], "genres": anime["genres"]})
        for anime in anime_data
    ]

    document_store.write_documents(docs)
    document_store.update_embeddings(retriever)

    return retriever, document_store