# LLM Safety API Service

A low-latency Jailbreak Detection API Service using `meta-llama/Llama-Guard-3-1B`.

## Requirements

- Docker
- Hugging Face Access Token (with access to `meta-llama/Llama-Guard-3-1B`)

## Quick Start (Docker)

### GPU（NVIDIA）
要讓容器用到 GPU，你需要：
- Windows + WSL2（你已在用 WSL）
- 安裝最新 NVIDIA 驅動（支援 WSL / CUDA）
- Docker Desktop 啟用 WSL2 integration
- Docker Desktop 支援 NVIDIA GPU（通常需搭配 `nvidia-container-toolkit`/Docker Desktop 的 GPU 支援）

先用這個確認容器能看到 GPU：
```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

**注意：** 即使沒有 GPU，`--gpus all` 也不會報錯，容器會自動 fallback 到 CPU。代碼會自動檢測 `torch.cuda.is_available()`，如果沒有 GPU 就會用 CPU 運行。

### Option 1: Bake Model into Image (Recommended for Submission)
This will download the model during the build process, so the final image can run anywhere without an internet connection or token.

1.  **Build with Token:**
    Pass your token using a Docker BuildKit secret named `HF_TOKEN` (recommended; keeps token out of image layers).
    ```bash
    # PowerShell
    $env:DOCKER_BUILDKIT=1
    $env:HF_TOKEN="your_hf_token_here"
    docker build --secret id=HF_TOKEN,env=HF_TOKEN -t llm-safety-service .
    ```

2.  **Run:**
    ```bash
    # CPU（沒有 GPU 時用這個）
    docker run -p 8000:8000 --env-file .env llm-safety-service

    # GPU（有 GPU 時加 --gpus all，沒有 GPU 也會自動 fallback 到 CPU）
    docker run -p 8000:8000 --gpus all --env-file .env llm-safety-service
    ```

**Note (WSL):** If `curl http://localhost:8000/health` fails, try forcing IPv4:
```bash
curl http://127.0.0.1:8000/health
```

### Option 2: Download at Runtime
If you build without the token/secret, the image will still build. The container will download the model when it starts (requires token at runtime).

1.  **Build:**
    ```bash
    docker build -t llm-safety-service .
    ```

2.  **Run with Token:**
    ```bash
    # 如果 8000 port 被佔用，先清理舊容器：
    docker ps -a | grep llm-safety | awk '{print $1}' | xargs -r docker rm -f

    # Option A: pass a single env var
    docker run -p 8000:8000 -e HF_TOKEN=your_hf_token_here llm-safety-service

    # Option B: use your local .env file (recommended)
    # CPU（沒有 GPU 時用這個）
    docker run -p 8000:8000 --env-file .env llm-safety-service

    # GPU（有 GPU 時加 --gpus all，沒有 GPU 也會自動 fallback 到 CPU，不會報錯）
    docker run -p 8000:8000 --gpus all --env-file .env llm-safety-service

    # 或者用 --name 指定名稱，方便管理：
    docker run --name llm-safety -p 8000:8000 --gpus all --env-file .env llm-safety-service
    ```


## Evaluation

Run the evaluation script against the running Docker container:
```bash
python evaluate.py --url http://localhost:8000
```
