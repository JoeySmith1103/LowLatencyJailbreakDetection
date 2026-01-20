FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Install system dependencies
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src /app/src

# Copy evaluate scripts and test data
COPY evaluate.py /app/evaluate.py
COPY evaluate_cache.py /app/evaluate_cache.py
COPY LLMSafetyAPIService_data.json /app/LLMSafetyAPIService_data.json

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]

