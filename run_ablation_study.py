"""
Comprehensive Ablation Study for Low-Latency Jailbreak Detection.

This script runs a complete ablation study comparing:
1. Baseline (no optimization)
2. Stopping Criteria only
3. Embedding Fast Path only
4. Full (Stopping + Embedding)

It also includes:
- Token generation analysis (why stopping criteria works)
- Statistical analysis (mean, std across multiple runs)
- Visualization

Usage:
    python run_ablation_study.py --data LLMSafetyAPIService_data.json --runs 3
"""
import os
import json
import time
import argparse
import statistics
from typing import List, Dict, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict

# For visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class SingleRunResult:
    """Result from a single evaluation run."""
    mode: str
    latencies: List[float] = field(default_factory=list)
    predictions: List[str] = field(default_factory=list)
    ground_truths: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    tokens_generated: List[int] = field(default_factory=list)
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0
    
    @property
    def std_latency(self) -> float:
        return statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0
    
    @property
    def avg_tokens(self) -> float:
        valid_tokens = [t for t in self.tokens_generated if t is not None]
        return statistics.mean(valid_tokens) if valid_tokens else 0
    
    @property
    def accuracy(self) -> float:
        correct = sum(1 for p, g in zip(self.predictions, self.ground_truths) if p == g)
        return correct / len(self.predictions) if self.predictions else 0
    
    def compute_metrics(self) -> Dict:
        """Compute precision, recall, F1."""
        tp = fp = tn = fn = 0
        for pred, gt in zip(self.predictions, self.ground_truths):
            if gt == "unsafe" and pred == "unsafe":
                tp += 1
            elif gt == "safe" and pred == "unsafe":
                fp += 1
            elif gt == "safe" and pred == "safe":
                tn += 1
            elif gt == "unsafe" and pred == "safe":
                fn += 1
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": self.accuracy
        }
    
    def layer_distribution(self) -> Dict[str, int]:
        """Count layer distribution."""
        dist = defaultdict(int)
        for layer in self.layers:
            dist[layer] += 1
        return dict(dist)


@dataclass
class AggregatedResult:
    """Aggregated results across multiple runs."""
    mode: str
    runs: List[SingleRunResult] = field(default_factory=list)
    
    @property
    def avg_latency(self) -> float:
        return statistics.mean([r.avg_latency for r in self.runs])
    
    @property
    def std_latency(self) -> float:
        """Standard deviation across runs."""
        if len(self.runs) <= 1:
            return 0
        return statistics.stdev([r.avg_latency for r in self.runs])
    
    @property
    def avg_tokens(self) -> float:
        valid = [r.avg_tokens for r in self.runs if r.avg_tokens > 0]
        return statistics.mean(valid) if valid else 0
    
    @property
    def avg_accuracy(self) -> float:
        return statistics.mean([r.accuracy for r in self.runs])
    
    def avg_metrics(self) -> Dict:
        """Average metrics across runs."""
        all_metrics = [r.compute_metrics() for r in self.runs]
        avg = {}
        for key in all_metrics[0].keys():
            avg[key] = statistics.mean([m[key] for m in all_metrics])
        return avg


def load_data(data_path: str) -> List[Dict]:
    """Load test data."""
    with open(data_path, 'r') as f:
        return json.load(f)


def reset_service():
    """Reset the singleton service for fresh initialization."""
    from src.services.safety_model import LlamaGuardService
    LlamaGuardService._instance = None


def run_single_evaluation(data: List[Dict], mode: str, threshold: float = 0.60) -> SingleRunResult:
    """
    Run a single evaluation with the specified mode.
    """
    # Set environment variables
    os.environ["OPTIMIZATION_MODE"] = mode
    os.environ["EMBEDDING_THRESHOLD"] = str(threshold)
    
    # Reset and reinitialize service
    reset_service()
    from src.services.safety_model import LlamaGuardService
    service = LlamaGuardService()
    
    result = SingleRunResult(mode=mode)
    
    for item in data:
        text = item['text']
        ground_truth = item['label']
        
        start_time = time.time()
        detailed = service.predict_detailed(text)
        latency_ms = (time.time() - start_time) * 1000
        
        result.latencies.append(latency_ms)
        result.predictions.append(detailed['final_label'])
        result.ground_truths.append(ground_truth)
        result.layers.append(detailed['layer_used'])
        result.tokens_generated.append(detailed.get('tokens_generated'))
    
    return result


def run_ablation_study(data: List[Dict], modes: List[str], num_runs: int = 3, 
                       threshold: float = 0.60, verbose: bool = True) -> Dict[str, AggregatedResult]:
    """
    Run complete ablation study across all modes.
    """
    results = {}
    
    for mode in modes:
        if verbose:
            print(f"\n{'='*60}")
            print(f"EVALUATING MODE: {mode.upper()}")
            print(f"{'='*60}")
        
        aggregated = AggregatedResult(mode=mode)
        
        for run_idx in range(num_runs):
            if verbose:
                print(f"  Run {run_idx + 1}/{num_runs}...", end=" ")
            
            run_result = run_single_evaluation(data, mode, threshold)
            aggregated.runs.append(run_result)
            
            if verbose:
                print(f"Latency: {run_result.avg_latency:.2f}ms, Accuracy: {run_result.accuracy*100:.1f}%")
        
        results[mode] = aggregated
        
        if verbose:
            print(f"  → Average: {aggregated.avg_latency:.2f}ms (±{aggregated.std_latency:.2f})")
    
    return results


def print_ablation_results(results: Dict[str, AggregatedResult], baseline_mode: str = "baseline"):
    """Print comprehensive ablation study results."""
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)
    
    baseline = results.get(baseline_mode)
    baseline_latency = baseline.avg_latency if baseline else None
    
    # Table header
    print(f"\n{'Mode':<20} {'Latency (ms)':<18} {'Improvement':<15} {'Accuracy':<12} {'Precision':<12} {'Recall':<10}")
    print("-" * 90)
    
    for mode, result in results.items():
        metrics = result.avg_metrics()
        latency_str = f"{result.avg_latency:.2f} (±{result.std_latency:.2f})"
        
        if baseline_latency and mode != baseline_mode:
            improvement = (baseline_latency - result.avg_latency) / baseline_latency * 100
            improvement_str = f"-{improvement:.1f}%"
        else:
            improvement_str = "-"
        
        print(f"{mode:<20} {latency_str:<18} {improvement_str:<15} "
              f"{metrics['accuracy']*100:<12.1f} {metrics['precision']*100:<12.1f} {metrics['recall']*100:<10.1f}")
    
    print("-" * 90)


def print_token_analysis(results: Dict[str, AggregatedResult]):
    """Print token generation analysis."""
    print("\n" + "=" * 70)
    print("TOKEN GENERATION ANALYSIS")
    print("=" * 70)
    print("\nThis shows WHY stopping criteria is effective:")
    print("- Baseline generates multiple tokens to complete the response")
    print("- With stopping criteria, we stop at the first 'safe'/'unsafe' token")
    print()
    
    print(f"{'Mode':<20} {'Avg Tokens Generated':<25} {'Note':<30}")
    print("-" * 70)
    
    for mode, result in results.items():
        avg_tokens = result.avg_tokens
        
        if "stopping" in mode or mode == "full":
            note = "Stops early at safe/unsafe"
        else:
            note = "Full generation"
        
        if avg_tokens > 0:
            print(f"{mode:<20} {avg_tokens:<25.2f} {note:<30}")
        else:
            print(f"{mode:<20} {'N/A (embedding only)':<25} {note:<30}")
    
    print("-" * 70)
    
    # Calculate token reduction
    baseline_tokens = results.get("baseline", AggregatedResult("")).avg_tokens
    stopping_tokens = results.get("stopping", AggregatedResult("")).avg_tokens
    
    if baseline_tokens > 0 and stopping_tokens > 0:
        reduction = (baseline_tokens - stopping_tokens) / baseline_tokens * 100
        print(f"\n→ Token reduction with stopping criteria: {reduction:.1f}%")
        print(f"  ({baseline_tokens:.1f} tokens → {stopping_tokens:.1f} tokens)")


def print_layer_distribution(results: Dict[str, AggregatedResult]):
    """Print layer distribution analysis."""
    print("\n" + "=" * 70)
    print("LAYER DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print("\nShows how many queries are handled by each layer:")
    print()
    
    for mode, result in results.items():
        if not result.runs:
            continue
        
        # Aggregate layer distribution across runs
        total_dist = defaultdict(int)
        total_queries = 0
        for run in result.runs:
            dist = run.layer_distribution()
            for layer, count in dist.items():
                total_dist[layer] += count
                total_queries += count
        
        print(f"{mode.upper()}:")
        for layer, count in sorted(total_dist.items()):
            pct = count / total_queries * 100 if total_queries > 0 else 0
            print(f"  {layer:<12}: {count:>5} ({pct:>5.1f}%)")
        print()


def plot_ablation_results(results: Dict[str, AggregatedResult], output_dir: str = "."):
    """Generate ablation study visualizations."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots")
        return
    
    modes = list(results.keys())
    latencies = [results[m].avg_latency for m in modes]
    latency_stds = [results[m].std_latency for m in modes]
    accuracies = [results[m].avg_accuracy * 100 for m in modes]
    
    # Plot 1: Latency comparison with error bars
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    bars = ax.bar(modes, latencies, yerr=latency_stds, capsize=5, 
                  color=colors[:len(modes)], edgecolor='black', linewidth=1)
    
    ax.set_ylabel('Average Latency (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Ablation Study: Latency Comparison', fontsize=14, fontweight='bold')
    ax.set_xlabel('Optimization Configuration', fontsize=12)
    
    # Add value labels
    for bar, lat, std in zip(bars, latencies, latency_stds):
        height = bar.get_height()
        ax.annotate(f'{lat:.1f}ms',
                    xy=(bar.get_x() + bar.get_width() / 2, height + std + 0.5),
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement percentages
    baseline_lat = latencies[0] if modes[0] == "baseline" else latencies[0]
    for i, (bar, lat) in enumerate(zip(bars, latencies)):
        if i > 0:
            improvement = (baseline_lat - lat) / baseline_lat * 100
            ax.annotate(f'-{improvement:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, lat / 2),
                        ha='center', va='center', fontsize=10, color='white', fontweight='bold')
    
    ax.set_ylim(0, max(latencies) * 1.3)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_latency.png', dpi=150)
    print(f"Saved: {output_dir}/ablation_latency.png")
    plt.close()
    
    # Plot 2: Latency vs Accuracy trade-off
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    x = range(len(modes))
    width = 0.35
    
    bars1 = ax1.bar([i - width/2 for i in x], latencies, width, label='Latency (ms)', 
                    color='steelblue', edgecolor='black')
    ax1.set_ylabel('Latency (ms)', color='steelblue', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='steelblue')
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i + width/2 for i in x], accuracies, width, label='Accuracy (%)', 
                    color='coral', edgecolor='black')
    ax2.set_ylabel('Accuracy (%)', color='coral', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='coral')
    ax2.set_ylim(90, 100)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(modes)
    ax1.set_xlabel('Optimization Configuration', fontsize=12)
    ax1.set_title('Ablation Study: Latency vs Accuracy', fontsize=14, fontweight='bold')
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend([bars1, bars2], ['Latency (ms)', 'Accuracy (%)'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ablation_tradeoff.png', dpi=150)
    print(f"Saved: {output_dir}/ablation_tradeoff.png")
    plt.close()
    


def main():
    parser = argparse.ArgumentParser(description="Run Ablation Study")
    parser.add_argument("--data", default="LLMSafetyAPIService_data.json", help="Path to test data")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per mode")
    parser.add_argument("--threshold", type=float, default=0.60, help="Embedding threshold for modes with embedding")
    parser.add_argument("--output-dir", default=".", help="Output directory for plots")
    parser.add_argument("--modes", type=str, default="baseline,stopping,embedding,full",
                        help="Comma-separated modes to test")
    args = parser.parse_args()
    
    modes = [m.strip() for m in args.modes.split(",")]
    
    print("=" * 70)
    print("ABLATION STUDY: LOW-LATENCY JAILBREAK DETECTION")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Modes: {modes}")
    print(f"Runs per mode: {args.runs}")
    print(f"Embedding threshold: {args.threshold}")
    print("=" * 70)
    
    # Load data
    data = load_data(args.data)
    print(f"Loaded {len(data)} test samples")
    
    # Run ablation study
    results = run_ablation_study(data, modes, num_runs=args.runs, 
                                  threshold=args.threshold, verbose=True)
    
    # Print results
    print_ablation_results(results)
    print_token_analysis(results)
    print_layer_distribution(results)
    
    # Generate plots
    if HAS_MATPLOTLIB:
        plot_ablation_results(results, args.output_dir)
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    baseline_lat = results.get("baseline", AggregatedResult("")).avg_latency
    
    print("\nKey Findings:")
    
    if "stopping" in results and baseline_lat > 0:
        stopping_lat = results["stopping"].avg_latency
        improvement = (baseline_lat - stopping_lat) / baseline_lat * 100
        print(f"1. Stopping Criteria alone: -{improvement:.1f}% latency")
    
    if "embedding" in results and baseline_lat > 0:
        embedding_lat = results["embedding"].avg_latency
        improvement = (baseline_lat - embedding_lat) / baseline_lat * 100
        print(f"2. Embedding Fast Path alone: -{improvement:.1f}% latency")
    
    if "full" in results and baseline_lat > 0:
        full_lat = results["full"].avg_latency
        improvement = (baseline_lat - full_lat) / baseline_lat * 100
        print(f"3. Full (Stopping + Embedding): -{improvement:.1f}% latency")
    
    print("\n✅ Ablation study complete!")


if __name__ == "__main__":
    main()
