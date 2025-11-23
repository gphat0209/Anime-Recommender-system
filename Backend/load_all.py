from database_loader.load_postgres import load_postgres_data
from database_loader.load_qdrant import load_qdrant_data

if __name__ == "__main__":
    print("🔄 Loading PostgreSQL...")
    load_postgres_data()

    print("🔄 Loading Qdrant...")
    load_qdrant_data()

    print("✅ DONE.")
