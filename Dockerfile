FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
ENV PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# --- BAKE MODEL INTO IMAGE ---
# Copy download script separately to leverage caching
COPY src/download_model.py .
# Use Docker BuildKit secrets to securely pass the HF token during build.
# This keeps the image clean and avoids security warnings.
RUN --mount=type=secret,id=HF_TOKEN \
    export HF_TOKEN=$(cat /run/secrets/HF_TOKEN) && \
    python3 download_model.py
# -----------------------------
# -----------------------------

# Copy source code (this changes frequently, so keep it late)
COPY src /app/src

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
