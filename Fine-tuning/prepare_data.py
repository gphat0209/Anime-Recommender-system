import pandas as pd
import numpy as np
import json, os
from sklearn.utils import shuffle
import math
from sqlalchemy import create_engine

from dotenv import load_dotenv
import os
load_dotenv()

database_url = os.getenv("DATABASE_URL")

engine = create_engine(database_url)
df = pd.read_sql("SELECT * FROM details", engine)

# df = pd.read_csv("data/preprocessed_anime_data.csv")

short_df = df[['sypnosis', 'genres', 'label']]
shuffle_df = shuffle(short_df, random_state=42).reset_index(drop=True)

train_length = math.floor(0.7*len(shuffle_df))
valid_length = math.floor(0.9*len(shuffle_df))

train_df = shuffle_df.iloc[:train_length]
# valid_df = shuffle_df.iloc[train_length:valid_length]
# test_df = shuffle_df.iloc[valid_length:]

test_df = shuffle_df.iloc[train_length:]

data_dir = 'training_data/'
os.makedirs(data_dir, exist_ok=True)

def get_training_data(df, path):
    # ánh xạ nhãn sang số
    label_map = {
        "Top-tier": 0,
        "Worthwhile": 1,
        "Watchable": 2,
        "Terrible": 3
    }

    data = []
    for _, row in df.iterrows():
        text = f"Synopsis: {row['sypnosis']}\nGenres: {row['genres']}"
        label = label_map[row['label']]
        
        data.append({
            "text": text,
            "label": label
        })
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_training_data_t5(df, path):
    """
    Tạo dữ liệu huấn luyện cho mô hình T5 (encoder-decoder).
    Dạng: {"input": "...", "output": "..."} thay vì text/label số.
    """
    data = []

    for _, row in df.iterrows():
        synopsis = row["sypnosis"]
        genres = row["genres"]
        label = row["label"]

        # ✨ Instruction prompt dạng natural language
        prompt = (
            "You are an expert anime critic.\n"
            "Your task is to evaluate an anime based on its synopsis and genre information.\n"
            "Consider the storytelling quality, originality, emotional depth, and audience appeal.\n\n"
            "Classify the anime into one of the following categories:\n"
            "- Top-tier: A masterpiece with excellent storytelling and execution.\n"
            "- Worthwhile: High-quality and enjoyable, though not perfect.\n"
            "- Watchable: Average, with some flaws but still entertaining.\n"
            "- Terrible: Poorly made or uninteresting.\n\n"
            f"Synopsis: {synopsis}\n"
            f"Genres: {genres}\n\n"
            "Respond only with one of the four labels: Top-tier, Worthwhile, Watchable, or Terrible."
        )

        # T5 mong output là văn bản
        output = label

        data.append({
            "input": prompt,
            "output": output
        })

    # Lưu file JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# get_training_data_t5(train_df, data_dir+'train.json')
get_training_data(shuffle_df, data_dir+'full.json')
# get_training_data_t5(test_df, data_dir+'test.json')
