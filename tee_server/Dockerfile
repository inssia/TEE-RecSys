# Simple Dockerfile for Phala Cloud TEE Deployment
FROM python:3.11-slim

WORKDIR /app

# Force logs to stdout immediately (avoid silent crashes!)
ENV PYTHONUNBUFFERED=1

# Install standard packages
COPY tee-requirements.txt .
RUN pip install --no-cache-dir -r tee-requirements.txt

# Use CPU-only torch to save space/RAM; version must match training
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu

# Create model directory
RUN mkdir -p /app/model

# Copy application code
# Model weights are part of TEE image for attestation
COPY tee_scorer_app.py .
COPY model_weights/relevance_estimator.pt /app/model/relevance_estimator.pt

# Environment variables
ENV PORT=4768
ENV EMBEDDING_DIM=64
ENV MODEL_PATH=/app/model/relevance_estimator.pt
ENV TEE_ENV=phala_cloud

# Expose port
EXPOSE 4768

# Using gunicorn for prod in TEE
CMD ["gunicorn", "--bind", "0.0.0.0:4768", "--workers", "2", "--threads", "2", "tee_scorer_app:app"]

