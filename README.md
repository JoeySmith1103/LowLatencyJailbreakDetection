# LLM Safety API Service

A low-latency Jailbreak Detection API Service using `meta-llama/Llama-Guard-3-1B`.

## Requirements

- Docker
- Hugging Face Access Token (with access to `meta-llama/Llama-Guard-3-1B`)

## Quick Start (Docker)

### Option 1: Bake Model into Image (Recommended for Submission)
This will download the model during the build process, so the final image can run anywhere without an internet connection or token.

1.  **Build with Token:**
    Pass your token using `--build-arg`.
    ```bash
    docker build --build-arg HF_TOKEN=your_hf_token_here -t llm-safety-service .
    ```

2.  **Run:**
    ```bash
    docker run -p 8000:8000 llm-safety-service
    ```

### Option 2: Download at Runtime
If you build without the token, the container will try to download the model when it starts. You must provide the token at runtime.

1.  **Build:**
    ```bash
    docker build -t llm-safety-service .
    ```

2.  **Run with Token:**
    ```bash
    docker run -p 8001:8000 -e HF_TOKEN=your_hf_token_here llm-safety-service
    ```


## Evaluation

Run the evaluation script against the running Docker container:
```bash
python evaluate.py --url http://localhost:8000
```
