from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient
from langchain_community.embeddings import GPT4AllEmbeddings
from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from sqlalchemy import create_engine, text
import re
import joblib
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate

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
    
MODEL_PATH = "../Fine-tuning/anime_model/lr_anime_classifier.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")

clf = joblib.load(MODEL_PATH)
print("✅ Loaded classifier:", MODEL_PATH)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



# === Request schema ===
class AnimeEvalRequest(BaseModel):
    synopsis: str
    genres: Optional[List[str]] = None

# === Response schema ===
class AnimeEvalResponse(BaseModel):
    label: str
    confidence: float
    label_id: int
    comment: str

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

chat_template = """
You are an AI anime critic assistant.
You will receive an anime's synopsis, genres, and a model's predicted quality label. 
Your job is to write a short, natural evaluation starting with the predicted rating, 
then briefly explain the reasoning in a human-like way.

--- Context ---
Synopsis: {synopsis}
Genres: {genres}
Model Prediction: {label} (Confidence: {confidence})
--- End Context ---

Guidelines:
- Start your response **by clearly stating the anime’s predicted quality** (e.g., “This anime is rated as ‘Worthwhile.’”)
- Follow up with a few sentences explaining *why* this rating makes sense, referring naturally to the synopsis or genres.
- If confidence < 0.5, express gentle uncertainty (e.g., “It might be...”, “It could go either way...”).
- If confidence ≥ 0.7, sound confident and decisive.
- Write in fluent, natural English — 3 to 5 sentences max.
- Keep it friendly, concise, and appropriate for anime fans.
- Do not mention anything about AI models or classifiers.
"""

prompt = ChatPromptTemplate.from_template(chat_template)
# === Endpoint chính ===
@app.post("/anime/evaluate", response_model=AnimeEvalResponse)
def evaluate_anime(req: AnimeEvalRequest):
    text = req.synopsis.strip()
    if req.genres:
        text += "\nGenres: " + ", ".join(req.genres)

    emb = embeddings.embed_query(text)
    pred_label = clf.predict([emb])[0]
    pred_proba = max(clf.predict_proba([emb])[0])

    label_map = {0: "Top-tier", 1: "Worthwhile", 2: "Watchable", 3: "Terrible"}
    label_name = label_map[int(pred_label)]

    # Gọi LLM tạo phản hồi tự nhiên
    llm_response = llm.invoke(
        prompt.format(
            synopsis=req.synopsis,
            genres=", ".join(req.genres or []),
            label=label_name,
            confidence=round(pred_proba, 3)
        )
    )

    print("LLM response:", llm_response.content.strip())

    print({
        "label_id": int(pred_label),
        "label": label_name,
        "confidence": round(pred_proba, 3),
        "comment": llm_response.content.strip(),
    })

    return {
        "label_id": int(pred_label),
        "label": label_name,
        "confidence": round(pred_proba, 3),
        "comment": llm_response.content.strip(),
    }
database_url = os.getenv("DATABASE_URL")
engine = create_engine(database_url)

@app.get("/anime/genres")
def get_unique_genres():
    """Lấy danh sách thể loại duy nhất từ bảng details"""
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT genres FROM details WHERE genres IS NOT NULL")).fetchall()

    # Lấy tất cả genre và làm sạch
    genres = []
    for (g,) in rows:
        if g:
            genres.extend(re.split(r',\s*', g.strip()))

    # Loại bỏ trùng và sắp xếp
    unique_genres = sorted(set([g.strip() for g in genres if g.strip()]))

    return {"genres": unique_genres}

