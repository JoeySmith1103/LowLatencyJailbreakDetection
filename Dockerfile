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

# --- BAKE MODEL INTO IMAGE ---
# Copy download script separately to leverage caching
COPY src/download_model.py .
# Optionally download the model during build.
# If you pass a BuildKit secret named HF_TOKEN, the model will be cached in the image.
# Otherwise the image will still build, and the model can be downloaded at runtime (requires env HF_TOKEN).
RUN --mount=type=secret,id=HF_TOKEN \
    bash -lc 'if [ -f /run/secrets/HF_TOKEN ]; then export HF_TOKEN="$(cat /run/secrets/HF_TOKEN)"; python3 download_model.py; else echo "HF_TOKEN secret not provided; skipping model download at build time."; fi'
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
