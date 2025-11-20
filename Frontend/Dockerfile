# 1️⃣ Base image — Python 3.10+ để tương thích Streamlit
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy toàn bộ mã nguồn vào container
COPY . /app

# 4️⃣ Cài đặt dependencies
RUN pip install --no-cache-dir streamlit pandas requests

# 5️⃣ Expose cổng mặc định Streamlit
EXPOSE 8501

# 6️⃣ Chạy Streamlit app
#   --server.port: cổng 8501
#   --server.address: 0.0.0.0 để cho phép truy cập từ ngoài container
CMD ["streamlit", "run", "frontend.py", "--server.port=8501", "--server.address=0.0.0.0"]
