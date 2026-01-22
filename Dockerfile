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

# Copy experiment scripts and test data
COPY analyze_threshold.py /app/analyze_threshold.py
COPY run_ablation_study.py /app/run_ablation_study.py
COPY LLMSafetyAPIService_data.json /app/LLMSafetyAPIService_data.json

# Pre-download model during build
ARG HF_TOKEN
RUN test -n "$HF_TOKEN" || (echo "ERROR: HF_TOKEN build arg is required" && exit 1)
ENV HF_TOKEN=${HF_TOKEN}
RUN python3 /app/src/download_model.py

# Expose port
EXPOSE 8001

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8001"]

