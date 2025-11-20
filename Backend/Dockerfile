# ==============================
# 1️⃣ Base image: Python
# ==============================
FROM python:3.10-slim

# ==============================
# 2️⃣ Set working directory
# ==============================
WORKDIR /app

# ==============================
# 3️⃣ Copy dependency files
# ==============================
COPY requirements.txt .

# ==============================
# 4️⃣ Install dependencies
# ==============================
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ==============================
# 5️⃣ Copy source code
# ==============================
COPY . .

# ==============================
# 6️⃣ Environment variables
# ==============================
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV QDRANT_PATH=/app/database/qdrant_anime_db
ENV GOOGLE_API_KEY=${GOOGLE_API_KEY}

# ==============================
# 7️⃣ Expose port
# ==============================
EXPOSE 8080

# ==============================
# 8️⃣ Run FastAPI
# ==============================
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
