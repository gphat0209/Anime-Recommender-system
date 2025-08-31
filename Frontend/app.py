from sklearn.neighbors import NearestNeighbors
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from utils import load_document_store
# from pyngrok import ngrok

app = FastAPI()
templates = Jinja2Templates(directory='templates')  

model_path = "./anime_deberta_model/checkpoint-1530"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.eval()


labels = ['Top-tier', 'Worthwhile', 'Watchable', 'Terrible']

class AnimeInput(BaseModel):
    synopsis: str  
    genres: str

class QueryInput(BaseModel):
    sypnosis: str

retriever, document_store = load_document_store("./training_data/short_file.json")

def find_similar_anime(user_synopsis: str, top_k=5):
    results = retriever.retrieve(query=user_synopsis, top_k=top_k)
    return [
        {
            "title": doc.meta["title"],
            "genres": doc.meta["genres"],
        }
        for doc in results
    ]

@app.get('/', response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse('chat.html', {'request': request})

@app.post('/predict')
async def predict(input: AnimeInput):
    prompt = (
        f"You are an intelligent anime recommender system. Your task is to evaluate how good an anime is based solely on its synopsis and genres.\n\n"
        f"<Anime Details>\n"
        f"Synopsis: {input.synopsis}\n"
        f"Genres: {input.genres}\n\n"
        "Based on this information, classify the anime strictly as one of the following: \"Top-tier\", \"Worthwhile\", \"Watchable\", or \"Terrible\"."
    )

    inputs = tokenizer(prompt, return_tensors='pt', truncation=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()

    return {
        'label': labels[prediction],
        'message': f"I think this anime is {labels[prediction]}"
    }

@app.post("/similar")
def get_similar_anime(input:QueryInput):
    return {"results: ": find_similar_anime(input.synopsis)} 

# public_url = ngrok.connect(8000)
# print("FastAPI app is live at:", public_url)

# # Run the FastAPI app
# import uvicorn
# uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
    
