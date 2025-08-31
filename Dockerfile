# Base image with Python
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-devel

# Set working directory in the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
# Copy the rest of the app code
COPY . .

# Expose port used by FastAPI
EXPOSE 8000

# Command to run the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
