"""
Comprehensive Threshold Analysis for Embedding Fast Path.

This script analyzes the trade-off between:
- Latency: Lower threshold → More embedding hits → Lower latency
- Precision: Higher threshold → Fewer false positives → Higher precision
- Recall: Lower threshold → Catch more unsafe queries → Higher recall (but may hurt precision)

The core insight:
- Embedding layer acts as a "fast path" for obvious unsafe queries
- Threshold controls HOW AGGRESSIVE the fast path is
- This is fundamentally a Latency vs Accuracy trade-off

Usage:
    # Start service first
    docker run --gpus all --env-file .env -p 8001:8001 -d --name jb-server lowlatency-jailbreak
    
    # Run threshold analysis
    docker exec jb-server python3 /app/analyze_threshold.py \
        --data /app/LLMSafetyAPIService_data.json \
        --api-url http://localhost:8001 \
        --case-study
"""
import json
import time
import argparse
import statistics
import requests
from typing import List, Dict
from dataclasses import dataclass, field

# For visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class PredictionResult:
    """Detailed prediction result for analysis."""
    text: str
    ground_truth: str
    predicted_label: str
    layer_used: str
    embedding_similarity: float = None
    matched_category: str = None
    matched_text: str = None
    tokens_generated: int = None
    latency_ms: float = 0.0
    
    @property
    def is_correct(self) -> bool:
        return self.predicted_label == self.ground_truth
    
    @property
    def is_true_positive(self) -> bool:
        return self.ground_truth == "unsafe" and self.predicted_label == "unsafe"
    
    @property
    def is_false_positive(self) -> bool:
        return self.ground_truth == "safe" and self.predicted_label == "unsafe"
    
    @property
    def is_true_negative(self) -> bool:
        return self.ground_truth == "safe" and self.predicted_label == "safe"
    
    @property
    def is_false_negative(self) -> bool:
        return self.ground_truth == "unsafe" and self.predicted_label == "safe"


@dataclass
class ThresholdAnalysis:
    """Analysis results for a specific threshold."""
    threshold: float
    results: List[PredictionResult] = field(default_factory=list)
    
    @property
    def total(self) -> int:
        return len(self.results)
    
    @property
    def tp(self) -> int:
        return sum(1 for r in self.results if r.is_true_positive)
    
    @property
    def fp(self) -> int:
        return sum(1 for r in self.results if r.is_false_positive)
    
    @property
    def tn(self) -> int:
        return sum(1 for r in self.results if r.is_true_negative)
    
    @property
    def fn(self) -> int:
        return sum(1 for r in self.results if r.is_false_negative)
    
    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.total if self.total > 0 else 0
    
    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0
    
    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0
    
    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0
    
    @property
    def avg_latency(self) -> float:
        latencies = [r.latency_ms for r in self.results]
        return statistics.mean(latencies) if latencies else 0
    
    @property
    def p50_latency(self) -> float:
        """Median latency (50th percentile)."""
        latencies = [r.latency_ms for r in self.results]
        return statistics.median(latencies) if latencies else 0
    
    @property
    def p90_latency(self) -> float:
        """90th percentile latency."""
        latencies = [r.latency_ms for r in self.results]
        return statistics.quantiles(latencies, n=10)[8] if len(latencies) >= 10 else max(latencies) if latencies else 0
    
    @property
    def p95_latency(self) -> float:
        """95th percentile latency."""
        latencies = [r.latency_ms for r in self.results]
        return statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies) if latencies else 0
    
    @property
    def p99_latency(self) -> float:
        """99th percentile (tail) latency."""
        latencies = [r.latency_ms for r in self.results]
        return statistics.quantiles(latencies, n=100)[98] if len(latencies) >= 100 else max(latencies) if latencies else 0
    
    @property
    def embedding_hit_rate(self) -> float:
        embedding_hits = sum(1 for r in self.results if r.layer_used == "embedding")
        return embedding_hits / self.total if self.total > 0 else 0
    
    @property
    def embedding_hits(self) -> List[PredictionResult]:
        return [r for r in self.results if r.layer_used == "embedding"]
    
    @property
    def llm_hits(self) -> List[PredictionResult]:
        return [r for r in self.results if r.layer_used == "llm"]


def load_data(data_path: str) -> List[Dict]:
    """Load test data."""
    with open(data_path, 'r') as f:
        return json.load(f)


def update_config(api_url: str, threshold: float) -> bool:
    """Update service configuration via HTTP API."""
    try:
        response = requests.post(
            f"{api_url}/admin/config",
            json={
                "optimization_mode": "full",
                "embedding_threshold": threshold
            },
            timeout=30
        )
        response.raise_for_status()
        config = response.json()
        print(f"  → Config updated: threshold={config['embedding_threshold']}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to update config: {e}")
        return False


def run_analysis_http(data: List[Dict], api_url: str, threshold: float) -> ThresholdAnalysis:
    """
    Run analysis via HTTP API for real-world performance.
    """
    # Update configuration
    if not update_config(api_url, threshold):
        raise RuntimeError(f"Failed to configure service for threshold: {threshold}")
    
    # Wait for service to stabilize
    time.sleep(2)
    
    analysis = ThresholdAnalysis(threshold=threshold)
    
    print(f"\nRunning analysis with threshold={threshold}...")
    
    for i, item in enumerate(data):
        text = item['text']
        ground_truth = item['label']
        
        # Measure latency including HTTP overhead
        start_time = time.time()
        try:
            response = requests.post(
                f"{api_url}/v1/detect/detailed",
                json={"text": text},
                timeout=30
            )
            response.raise_for_status()
            detailed = response.json()
            latency_ms = (time.time() - start_time) * 1000
            
            result = PredictionResult(
                text=text,
                ground_truth=ground_truth,
                predicted_label=detailed['label'],
                layer_used=detailed['layer'],
                embedding_similarity=detailed.get('embedding_similarity'),
                matched_category=detailed.get('matched_category'),
                matched_text=detailed.get('matched_text'),
                tokens_generated=detailed.get('tokens_generated'),
                latency_ms=latency_ms,
            )
            analysis.results.append(result)
        except Exception as e:
            print(f"  ✗ Error processing query: {e}")
            continue
        
        # Progress
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(data)} queries...")
    
    return analysis


def print_analysis_summary(analysis: ThresholdAnalysis):
    """Print summary of analysis results."""
    print("\n" + "=" * 70)
    print(f"THRESHOLD ANALYSIS: {analysis.threshold}")
    print("=" * 70)
    
    print("\n--- Performance Metrics ---")
    print(f"  Accuracy:   {analysis.accuracy * 100:.2f}%")
    print(f"  Precision:  {analysis.precision * 100:.2f}%")
    print(f"  Recall:     {analysis.recall * 100:.2f}%")
    print(f"  F1 Score:   {analysis.f1 * 100:.2f}%")
    
    print("\n--- Confusion Matrix ---")
    print(f"  TP (unsafe→unsafe): {analysis.tp}")
    print(f"  FP (safe→unsafe):   {analysis.fp}  ← False alarms (blocks safe content)")
    print(f"  TN (safe→safe):     {analysis.tn}")
    print(f"  FN (unsafe→safe):   {analysis.fn}  ← DANGEROUS (misses unsafe content)")
    
    print("\n--- Latency & Layer Distribution ---")
    print(f"  Avg Latency:        {analysis.avg_latency:.2f} ms")
    print(f"  Embedding Hit Rate: {analysis.embedding_hit_rate * 100:.1f}%")
    print(f"  Embedding Hits:     {len(analysis.embedding_hits)}")
    print(f"  LLM Hits:           {len(analysis.llm_hits)}")
    
    print("=" * 70)


def print_case_study(analysis: ThresholdAnalysis, top_n: int = 5):
    """Print detailed case study."""
    print("\n" + "=" * 70)
    print("CASE STUDY")
    print("=" * 70)
    
    # True Positives via Embedding (Success cases)
    tp_embedding = [r for r in analysis.embedding_hits if r.is_true_positive]
    print(f"\nTRUE POSITIVES via Embedding ({len(tp_embedding)} cases)")
    print("   (Successfully blocked by fast path)")
    for r in tp_embedding[:top_n]:
        print(f"\n   Query: \"{r.text[:60]}...\"" if len(r.text) > 60 else f"\n   Query: \"{r.text}\"")
        print(f"   Similarity: {r.embedding_similarity:.3f} | Matched: {r.matched_category}")
        if r.matched_text:
            print(f"   Matched Text: \"{r.matched_text[:50]}...\"" if len(r.matched_text) > 50 else f"   Matched Text: \"{r.matched_text}\"")
    
    # False Positives via Embedding (Problem cases)
    fp_embedding = [r for r in analysis.embedding_hits if r.is_false_positive]
    print(f"\nFALSE POSITIVES via Embedding ({len(fp_embedding)} cases)")
    print("   (Safe content incorrectly blocked by fast path)")
    for r in fp_embedding[:top_n]:
        print(f"\n   Query: \"{r.text[:60]}...\"" if len(r.text) > 60 else f"\n   Query: \"{r.text}\"")
        print(f"   Similarity: {r.embedding_similarity:.3f} | Matched: {r.matched_category}")
        if r.matched_text:
            print(f"   Matched Text: \"{r.matched_text[:50]}...\"" if len(r.matched_text) > 50 else f"   Matched Text: \"{r.matched_text}\"")
    
    # False Negatives (Dangerous cases)
    fn_cases = [r for r in analysis.results if r.is_false_negative]
    print(f"\nFALSE NEGATIVES ({len(fn_cases)} cases)")
    print("   (Unsafe content that was NOT detected - DANGEROUS)")
    for r in fn_cases[:top_n]:
        print(f"\n   Query: \"{r.text[:60]}...\"" if len(r.text) > 60 else f"\n   Query: \"{r.text}\"")
        if r.embedding_similarity:
            print(f"   Similarity: {r.embedding_similarity:.3f} | Layer: {r.layer_used}")
        else:
            print(f"   Layer: {r.layer_used}")
    
    # Interesting: High similarity but safe (near misses)
    safe_high_sim = [r for r in analysis.results 
                    if r.ground_truth == "safe" 
                    and r.embedding_similarity is not None 
                    and r.embedding_similarity > analysis.threshold - 0.1]
    safe_high_sim.sort(key=lambda x: x.embedding_similarity, reverse=True)
    
    print(f"\nNEAR MISSES (safe queries with high similarity)")
    print("   (These would be blocked if threshold was lower)")
    for r in safe_high_sim[:top_n]:
        print(f"\n   Query: \"{r.text[:60]}...\"" if len(r.text) > 60 else f"\n   Query: \"{r.text}\"")
        print(f"   Similarity: {r.embedding_similarity:.3f} | Threshold: {analysis.threshold}")
        print(f"   Blocked by embedding: {'Yes' if r.layer_used == 'embedding' else 'No'}")


def compare_thresholds(analyses: List[ThresholdAnalysis]):
    """Print comparison table across thresholds."""
    print("\n" + "=" * 90)
    print("THRESHOLD COMPARISON (via HTTP API)")
    print("=" * 90)
    
    print(f"\n{'Threshold':<12} {'Latency':<12} {'Emb Hit %':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'FP':<6} {'FN':<6}")
    print("-" * 90)
    
    for a in analyses:
        print(f"{a.threshold:<12.2f} {a.avg_latency:<12.2f} {a.embedding_hit_rate*100:<12.1f} "
              f"{a.precision*100:<12.1f} {a.recall*100:<12.1f} {a.f1*100:<12.1f} "
              f"{a.fp:<6} {a.fn:<6}")
    
    print("-" * 90)
    
    # Latency percentile breakdown
    print("\n" + "=" * 90)
    print("LATENCY PERCENTILE BREAKDOWN (ms)")
    print("=" * 90)
    print(f"\n{'Threshold':<12} {'P50':<12} {'P90':<12} {'P95':<12} {'P99':<12}")
    print("-" * 90)
    
    for a in analyses:
        print(f"{a.threshold:<12.2f} {a.p50_latency:<12.2f} {a.p90_latency:<12.2f} "
              f"{a.p95_latency:<12.2f} {a.p99_latency:<12.2f}")
    
    print("-" * 90)
    print("\n💡 P99 (tail latency) is critical for SLA - represents worst-case user experience")
    
    # Find best for each metric
    best_latency = min(analyses, key=lambda x: x.avg_latency)
    best_p99 = min(analyses, key=lambda x: x.p99_latency)
    best_precision = max(analyses, key=lambda x: x.precision)
    best_recall = max(analyses, key=lambda x: x.recall)
    best_f1 = max(analyses, key=lambda x: x.f1)
    
    print(f"\nBest Avg Latency: threshold={best_latency.threshold} ({best_latency.avg_latency:.2f}ms)")
    print(f"Best P99 Latency: threshold={best_p99.threshold} ({best_p99.p99_latency:.2f}ms)")
    print(f"Best Precision:   threshold={best_precision.threshold} ({best_precision.precision*100:.1f}%)")
    print(f"Best Recall:      threshold={best_recall.threshold} ({best_recall.recall*100:.1f}%)")
    print(f"Best F1:          threshold={best_f1.threshold} ({best_f1.f1*100:.1f}%)")


def plot_threshold_tradeoff(analyses: List[ThresholdAnalysis], output_dir: str = "."):
    """Generate trade-off visualization."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return
    
    thresholds = [a.threshold for a in analyses]
    latencies = [a.avg_latency for a in analyses]
    precisions = [a.precision * 100 for a in analyses]
    recalls = [a.recall * 100 for a in analyses]
    emb_hits = [a.embedding_hit_rate * 100 for a in analyses]
    
    # Plot 1: Latency vs Embedding Hit Rate
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Embedding Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Latency (ms)', color=color, fontsize=12, fontweight='bold')
    line1 = ax1.plot(thresholds, latencies, marker='o', color=color, linewidth=2, label='Latency')
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel('Embedding Hit Rate (%)', color=color, fontsize=12, fontweight='bold')
    line2 = ax2.plot(thresholds, emb_hits, marker='s', linestyle='--', color=color, linewidth=2, label='Embedding Hits')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Add annotations
    ax1.axvline(x=0.60, color='green', linestyle=':', alpha=0.7, label='Recommended (0.60)')
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.title('Threshold vs API Latency & Embedding Hit Rate', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig(f'{output_dir}/analysis_latency_vs_threshold.png', dpi=150)
    print(f"Saved: {output_dir}/analysis_latency_vs_threshold.png")
    plt.close()
    
    # Plot 2: Precision vs Recall Trade-off
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, marker='o', label='Precision', linewidth=2, color='forestgreen')
    ax.plot(thresholds, recalls, marker='s', label='Recall', linewidth=2, color='coral')
    
    # Highlight danger zone
    ax.fill_between(thresholds, recalls, 100, alpha=0.1, color='red', label='Missed unsafe (FN)')
    
    ax.set_xlabel('Embedding Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title('Threshold vs Precision & Recall Trade-off', fontsize=14, fontweight='bold')
    ax.set_ylim(85, 101)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/analysis_precision_recall.png', dpi=150)
    print(f"Saved: {output_dir}/analysis_precision_recall.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze Embedding Threshold Trade-offs via HTTP API")
    parser.add_argument("--data", default="LLMSafetyAPIService_data.json", help="Path to test data")
    parser.add_argument("--thresholds", type=str, default="0.50,0.55,0.60,0.65,0.70,0.75,0.80",
                        help="Comma-separated thresholds to test")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    parser.add_argument("--case-study", action="store_true", help="Print detailed case study")
    parser.add_argument("--api-url", default="http://localhost:8001", help="API base URL")
    args = parser.parse_args()
    
    # Parse thresholds
    thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
    
    print("=" * 70)
    print("EMBEDDING THRESHOLD ANALYSIS (HTTP API)")
    print("=" * 70)
    print(f"API URL: {args.api_url}")
    print(f"Data: {args.data}")
    print(f"Thresholds: {thresholds}")
    print("=" * 70)
    
    # Check API health
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        response.raise_for_status()
        print(f"✓ API is healthy: {response.json()}")
    except Exception as e:
        print(f"✗ Failed to connect to API: {e}")
        print("  Make sure the service is running with:")
        print("  docker run --gpus all --env-file .env -p 8001:8001 -d --name jb-server lowlatency-jailbreak")
        return
    
    # Load data
    data = load_data(args.data)
    print(f"Loaded {len(data)} test samples")
    
    # Run analysis for each threshold
    analyses = []
    for threshold in thresholds:
        analysis = run_analysis_http(data, args.api_url, threshold)
        analyses.append(analysis)
        print_analysis_summary(analysis)
        
        if args.case_study:
            print_case_study(analysis)
    
    # Compare all thresholds
    compare_thresholds(analyses)
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_threshold_tradeoff(analyses, args.output_dir)
    
    print("\n✅ Analysis complete!")
    print("\nNote: Latency includes HTTP request/response overhead.")
    print("      This represents real-world API performance.")


if __name__ == "__main__":
    main()
