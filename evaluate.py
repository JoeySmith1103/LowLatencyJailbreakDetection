import json
import time
import requests
import argparse
import statistics
from typing import List, Dict

def evaluate(api_url: str, data_path: str):
    print(f"Loading data from {data_path}...")
    with open(data_path, 'r') as f:
        data = json.load(f)

    latencies = []
    correct_count = 0
    total_count = len(data)
    
    # Warmup request to exclude cold start from metrics
    print("Warming up model...")
    try:
        requests.post(f"{api_url}/v1/detect", json={"text": "warmup"})
    except Exception:
        pass
    print("Warmup done.")
    
    print(f"Starting evaluation on {total_count} items...")
    
    for i, item in enumerate(data):
        text = item['text']
        ground_truth = item['label']
        
        start_time = time.time()
        try:
            response = requests.post(f"{api_url}/v1/detect", json={"text": text})
            response.raise_for_status()
            result = response.json()
            prediction = result['label']
        except Exception as e:
            print(f"Error on item {i}: {e}")
            prediction = "error"
        
        end_time = time.time()
        latency_ms = (end_time - start_time) * 1000
        latencies.append(latency_ms)
        
        if prediction == ground_truth:
            correct_count += 1
            
        print(f"[{i+1}/{total_count}] Label: {prediction} (GT: {ground_truth}) | Latency: {latency_ms:.2f}ms")

    accuracy = (correct_count / total_count) * 100
    avg_latency = statistics.mean(latencies)
    p99_latency = statistics.quantiles(latencies, n=100)[98] # approx P99

    print("\n" + "="*30)
    print("EVALUATION RESULTS")
    print("="*30)
    print(f"Total Requests: {total_count}")
    print(f"Accuracy:       {accuracy:.2f}%")
    print(f"Avg Latency:    {avg_latency:.2f} ms")
    print(f"P99 Latency:    {p99_latency:.2f} ms")
    print("="*30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8001", help="API URL")
    parser.add_argument("--data", default="/app/LLMSafetyAPIService_data.json", help="Path to test data")
    args = parser.parse_args()
    
    evaluate(args.url, args.data)
