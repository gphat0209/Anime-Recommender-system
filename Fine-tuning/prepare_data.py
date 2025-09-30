import pandas as pd
import numpy as np
import json, os
from sklearn.utils import shuffle
import math
from sqlalchemy import create_engine

engine = create_engine("postgresql+psycopg2://postgres:phatdeptrai123@localhost:5432/anime_info")
df = pd.read_sql("SELECT * FROM details", engine)

# df = pd.read_csv("data/preprocessed_anime_data.csv")

short_df = df[['sypnosis', 'genres', 'label']]
shuffle_df = shuffle(short_df, random_state=42).reset_index(drop=True)

train_length = math.floor(0.6*len(shuffle_df))
valid_length = math.floor(0.9*len(shuffle_df))

train_df = shuffle_df.iloc[:train_length]
valid_df = shuffle_df.iloc[train_length:valid_length]
test_df = shuffle_df.iloc[valid_length:]

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


get_training_data(train_df, data_dir+'train.json')
get_training_data(valid_df, data_dir+'valid.json')
get_training_data(test_df, data_dir+'test.json')
