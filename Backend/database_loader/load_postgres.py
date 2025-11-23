import psycopg2
import json
import os
from dotenv import load_dotenv

def load_postgres_data():
    load_dotenv()

    # POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
    # POSTGRES_DB = os.getenv("POSTGRES_DB", "anime_db")
    # POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
    # POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
    # POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

    # conn = psycopg2.connect(
    #     host=POSTGRES_HOST,
    #     database=POSTGRES_DB,
    #     user=POSTGRES_USER,
    #     password=POSTGRES_PASSWORD,
    #     port=POSTGRES_PORT
    # )

    DATABASE_URL = os.getenv("DATABASE_URL") 

    print("🔌Connecting to PostgreSQL using DATABASE_URL...")
    conn = psycopg2.connect(DATABASE_URL)

    cur = conn.cursor()

    # Create table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS anime (
        id INT PRIMARY KEY,
        name TEXT,
        type TEXT,
        episodes INT,
        aired TEXT,
        members INT,
        score FLOAT,
        label TEXT,
        synopsis TEXT,
        genres TEXT
    );
    """)
    conn.commit()

    print("Table 'anime' is ready.")

    # Load JSON
    json_path = "/app/output.json"
    with open(json_path, "r", encoding="utf-8") as f:
        anime_list = json.load(f)

    print(f"Loading {len(anime_list)} anime into PostgreSQL...")

    # Insert rows
    for anime in anime_list:
        try:
            cur.execute("""
                INSERT INTO anime (id, name, type, episodes, aired, members, score, label, synopsis, genres)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (id) DO NOTHING;
            """, (
                anime.get("ID"),
                anime.get("Name"),
                anime.get("Type"),
                int(anime.get("Episodes", 0)) if anime.get("Episodes") else None,
                anime.get("Aired"),
                anime.get("Members"),
                anime.get("Scores"),
                anime.get("Label"),
                anime.get("Sypnosis"),
                anime.get("Genres")
            ))
        except Exception as e:
            print(f"Error inserting {anime.get('Name')}: {e}")

    conn.commit()
    print("✨ PostgreSQL loading completed!")

    cur.close()
    conn.close()


# Allow standalone execution (optional)
if __name__ == "__main__":
    load_postgres_data()
