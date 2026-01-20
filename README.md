# LLM Safety API Service

A low-latency Jailbreak Detection API Service using `meta-llama/Llama-Guard-3-1B`.

## Optimization Strategies

本服務實作了 **3 層 Fast Path 架構** 來降低 Latency，同時維持高準確率：

```
Request → Layer 1 (Cache) → Layer 2 (Embedding) → Layer 3 (LLaMA Guard) → Response
```

### Layer 1: Exact Match Cache (System Level)

對相同的查詢進行快取，使用 MD5 hash 作為 cache key。

- **Latency**: ~0.6ms
- **實作位置**: `src/services/safety_model.py` - `_cache` dict

### Layer 2: Embedding Similarity (Architecture Level)

使用 `all-MiniLM-L6-v2` 計算 query 與 S1-S13 unsafe categories 的 cosine similarity。若 similarity > 0.85，直接判定為 unsafe，跳過 LLaMA Guard。

- **Latency**: ~3-5ms
- **實作位置**: `src/services/safety_model.py` - `_check_embedding_similarity()`

### Layer 3: LLaMA Guard + Custom Stopping Criteria (Inference Level)

最終 fallback 使用 LLaMA Guard 推論。實作 Custom Stopping Criteria，當偵測到 "safe" 或 "unsafe" token 時立即停止生成。

- **Latency**: ~17-19ms
- **實作位置**: `src/services/safety_model.py` - `SafetyStoppingCriteria` class

## Performance Comparison

### Test 1: Baseline vs Optimized (100 unique queries)

| Metric | Baseline | Optimized | Improvement |
|--------|----------|-----------|-------------|
| **Accuracy** | 95.00% | 98.00% | **+3%** |
| **Avg Latency** | 30.87 ms | 19.08 ms | **-38%** |
| **P99 Latency** | 42.61 ms | 21.82 ms | **-49%** |

### Test 2: Cache Effectiveness (300 queries, 3x repeated + shuffled)

| Metric | Value |
|--------|-------|
| **Total Requests** | 300 |
| **Unique Queries** | 100 |
| **Cache Hit Rate** | 66.7% |
| **Accuracy** | 98.00% |
| **Avg Latency** | **0.64 ms** |
| **P99 Latency** | **0.84 ms** |

> **測試環境**: NVIDIA RTX 4080, Docker with `--gpus all`

## Running Experiments

### Step 1: Build Docker Image
```bash
docker build -t lowlatency-jailbreak .
```

### Step 2: Run Server

**Baseline Mode (無優化)**：
```bash
docker run --gpus all --env-file .env -e OPTIMIZATION_MODE=baseline -p 8001:8001 -d --name jailbreak-server lowlatency-jailbreak
```

**Optimized Mode (3 層架構)**：
```bash
docker run --gpus all --env-file .env -e OPTIMIZATION_MODE=optimized -p 8001:8001 -d --name jailbreak-server lowlatency-jailbreak
```

### Step 3: Run Evaluation (inside Docker)
```bash
# 等待 server 啟動 (約 30-45 秒)
sleep 45

# 100 筆 unique queries
docker exec jailbreak-server python3 /app/evaluate.py

# 300 筆 (3x repeated + shuffled) - 測試 cache 效果
docker exec jailbreak-server python3 /app/evaluate_cache.py --repeats 3 --shuffle
```

### Step 4: Stop Server
```bash
docker stop jailbreak-server && docker rm jailbreak-server
```

## Requirements

- Docker
- NVIDIA GPU with CUDA support (optional, falls back to CPU)
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
    docker build -t lowlatency-jailbreak .
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
