FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for scikit-learn
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY conda.yaml /app/

# Install Python packages (use pre-built wheels)
RUN pip install --no-cache-dir \
    "setuptools<70" \
    "importlib-metadata<7.0" \
    pandas==2.0.3 \
    scikit-learn==1.7.2 \
    mlflow==2.9.2 \
    matplotlib==3.7.2 \
    seaborn==0.12.2 \
    joblib==1.3.2

# Copy all project files
COPY . /app/

# Expose MLflow serving port
EXPOSE 5000

# Entry point - serve the model
CMD ["python", "-m", "http.server", "5000"]
