FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy conda environment file
COPY conda.yaml /app/

# Create conda environment
RUN conda env create -f conda.yaml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "spam_detection_env", "/bin/bash", "-c"]

# Copy all project files
COPY . /app/

# Expose MLflow serving port
EXPOSE 5000

# Set environment variable
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Entry point to serve the model
CMD ["conda", "run", "--no-capture-output", "-n", "spam_detection_env", "mlflow", "models", "serve", "-m", "models/spam_detection_model.joblib", "-h", "0.0.0.0", "-p", "5000"]
