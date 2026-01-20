"""
Evaluation script with cache effectiveness testing.
Supports repeated queries with random shuffling to measure cache hit rate.
"""
import json
import time
import random
import requests
import argparse
import statistics
from typing import List, Dict


def load_data(data_path: str, repeats: int = 1, shuffle: bool = False) -> List[Dict]:
    """Load and optionally repeat/shuffle test data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if repeats > 1:
        # Repeat the data
        data = data * repeats
    
    if shuffle:
        random.shuffle(data)
    
    return data


def evaluate(api_url: str, data: List[Dict], verbose: bool = True):
    """Run evaluation and collect metrics."""
    latencies = []
    correct_count = 0
    total_count = len(data)
    
    # Layer statistics
    layer_counts = {"cache": 0, "embedding": 0, "llm": 0}
    
    # Warmup
    if verbose:
        print("Warming up model...")
    try:
        requests.post(f"{api_url}/v1/detect", json={"text": "warmup"})
    except Exception:
        pass
    if verbose:
        print("Warmup done.\n")
    
    for i, item in enumerate(data):
        text = item['text']
        ground_truth = item['label']
        
        start_time = time.time()
        try:
            response = requests.post(f"{api_url}/v1/detect", json={"text": text})
            response.raise_for_status()
            result = response.json()
            prediction = result['label']
            layer = result.get('layer', 'unknown')
        except Exception as e:
            if verbose:
                print(f"Error on item {i}: {e}")
            prediction = "error"
            layer = "error"
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if prediction == ground_truth:
            correct_count += 1
        
        # Count by layer
        if layer in layer_counts:
            layer_counts[layer] += 1
        
        # Status indicator based on server-provided layer
        layer_tag = f"[{layer.upper()}]"
            
        if verbose:
            print(f"[{i+1}/{total_count}] {layer_tag} {prediction} (GT: {ground_truth}) | {latency_ms:.2f}ms")

    # Calculate metrics
    accuracy = (correct_count / total_count) * 100
    avg_latency = statistics.mean(latencies)
    p99_latency = statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies)
    
    # Cache hit rate
    cache_hit_rate = (layer_counts["cache"] / total_count) * 100 if total_count > 0 else 0

    return {
        "total_requests": total_count,
        "layer_counts": layer_counts,
        "cache_hit_rate": cache_hit_rate,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "p99_latency": p99_latency,
        "latencies": latencies,
    }


def print_results(results: Dict):
    """Print evaluation results."""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total Requests:           {results['total_requests']}")
    print(f"Layer Distribution:")
    for layer, count in results['layer_counts'].items():
        pct = (count / results['total_requests']) * 100 if results['total_requests'] > 0 else 0
        print(f"  - {layer.upper():10s}: {count:4d} ({pct:.1f}%)")
    print(f"Cache Hit Rate:           {results['cache_hit_rate']:.1f}%")
    print("-" * 50)
    print(f"Accuracy:                 {results['accuracy']:.2f}%")
    print(f"Avg Latency:              {results['avg_latency']:.2f} ms")
    print(f"P99 Latency:              {results['p99_latency']:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache effectiveness evaluation")
    parser.add_argument("--url", default="http://localhost:8001", help="API URL")
    parser.add_argument("--data", default="/app/LLMSafetyAPIService_data.json", help="Path to test data")
    parser.add_argument("--repeats", type=int, default=1, help="Number of times to repeat data")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle data after repeating")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-request output")
    args = parser.parse_args()
    
    print(f"Loading data from {args.data}...")
    print(f"Repeats: {args.repeats}, Shuffle: {args.shuffle}")
    
    data = load_data(args.data, repeats=args.repeats, shuffle=args.shuffle)
    print(f"Total queries to evaluate: {len(data)}\n")
    
    results = evaluate(args.url, data, verbose=not args.quiet)
    print_results(results)
