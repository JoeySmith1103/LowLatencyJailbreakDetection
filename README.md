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


