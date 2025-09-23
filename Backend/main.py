from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from langchain_community.embeddings import GPT4AllEmbeddings


from dotenv import load_dotenv
import os
load_dotenv()

from services.search import search_anime_hybrid, search_anime_semantic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "database", "qdrant_anime_db")


# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
#embedding_model = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL_NAME, api_key = GOOGLE_API_KEY)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION")
embedding_model = GPT4AllEmbeddings()
qdrant = QdrantClient(path=DB_PATH)
collections = qdrant.get_collections()
print(collections)

app = FastAPI(title="CV Manager API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class SearchRequest(BaseModel):
    query_text: str

@app.get("/status")
def status():
    return {"status": "good"}

@app.get("/anime/search")
async def search(q:str):
    try:
        #results = search_anime_hybrid(
        results = search_anime_semantic(
            qdrant=qdrant,
            collection_name=COLLECTION_NAME,
            embedding_model=embedding_model,
            query_text=q,
        )
        return results
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")


