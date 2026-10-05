"""
Aggregated Results Generator for Subject-Verb Number Agreement
==============================================================
This script loads the ablation results from Zero, Mean, Resample, 
and ACDC analyses, computes comparative statistics, and generates 
a summary table and visualization for the final report.
"""

import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

def load_causal_scrubbing(filepath: Path) -> Optional[Dict]:
    """Load causal scrubbing validation results."""
    if not filepath.exists():
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def load_results(filepath: Path) -> List[Dict]:
    """Safely load ablation results from a JSON file."""
    if not filepath.exists():
        print(f"  [!] Warning: {filepath.name} not found at {filepath}")
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data.get("valid_ablations", [])

def compute_stats(name: str, records: List[Dict]) -> Optional[Dict]:
    """Compute summary statistics matching the report format."""
    if not records:
        return None
    
    drops = [r["performance_drop"] for r in records]
    max_idx = np.argmax(drops)
    top_record = records[max_idx]
    
    return {
        "Method": name,
        "Records": len(drops),
        "Mean Drop": np.mean(drops),
        "Median Drop": np.median(drops),
        "Std Dev": np.std(drops),
        "Min Drop": np.min(drops),
        "Max Drop": np.max(drops),
        "Top Head": f"L{top_record['layer']}H{top_record['head']}",
        "Top Head Drop": np.max(drops),
        "Positive Count": sum(1 for d in drops if d > 0),
        "Negative Count": sum(1 for d in drops if d < 0)
    }

def generate_markdown_table(stats_list: List[Dict]):
    """Print a cleanly formatted Markdown table for the report."""
    print("\n### Side-by-Side Comparison Summary")
    print("| Method | Records | Mean Drop | Median Drop | Std Dev | Min Drop | Max Drop | Top Head | Top Head Drop | Pos. Count | Neg. Count |")
    print("|--------|---------|-----------|-------------|---------|----------|----------|----------|---------------|------------|------------|")
    
    for s in stats_list:
        print(f"| **{s['Method']}** | {s['Records']} | {s['Mean Drop']:.4f} | {s['Median Drop']:.4f} | {s['Std Dev']:.4f} | {s['Min Drop']:.4f} | {s['Max Drop']:.4f} | **{s['Top Head']}** | {s['Top Head Drop']:.4f} | {s['Positive Count']} | {s['Negative Count']} |")
    print("\n")

def report_causal_scrubbing(data: Dict):
    """Print a cleanly formatted Markdown table for Causal Scrubbing."""
    if not data: return
    metrics = data.get("metrics", {})
    circuit_size = data.get("circuit_size", 0)
    
    base_clean = metrics.get("baseline_clean", 0)
    scrub_out = metrics.get("scrub_outside", 0)
    scrub_in = metrics.get("scrub_inside", 0)
    base_corr = metrics.get("baseline_corrupted", 0)
    
    denom = base_clean - base_corr
    faithfulness = (scrub_out - base_corr) / denom * 100 if denom != 0 else 0
    completeness = (base_clean - scrub_in) / denom * 100 if denom != 0 else 0

    print("### Circuit Validation (Causal Scrubbing)")
    print(f"**Circuit Size:** {circuit_size} heads\n")
    print(f"- **Faithfulness:** `{faithfulness:.2f}%` (Performance retained by circuit alone)")
    print(f"- **Completeness:** `{completeness:.2f}%` (Performance degraded by ablating circuit alone)\n")
    print("| Metric | Average Logit Diff |")
    print("|--------|--------------------|")
    print(f"| Baseline Clean | {base_clean:.4f} |")
    print(f"| Scrubbed Outside (Faithful) | {scrub_out:.4f} |")
    print(f"| Scrubbed Inside (Complete) | {scrub_in:.4f} |")
    print(f"| Baseline Corrupted | {base_corr:.4f} |\n")

def create_visualization(stats_list: List[Dict], output_dir: Path):
    """Create a grouped bar chart comparing Mean Drop vs Top Head Drop."""
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = [s["Method"] for s in stats_list]
    mean_drops = [s["Mean Drop"] for s in stats_list]
    max_drops = [s["Max Drop"] for s in stats_list]
    
    x = np.arange(len(methods))
    width = 0.35
    
    # Grouped bars
    bars1 = ax.bar(x - width/2, mean_drops, width, label='Mean Drop (Average Impact)', color='#3498db', edgecolor="black")
    bars2 = ax.bar(x + width/2, max_drops, width, label='Top Head Drop (Max Impact)', color='#e74c3c', edgecolor="black")
    
    # Styling
    ax.set_ylabel('Logit Difference Drop', fontsize=12, fontweight='bold')
    ax.set_title('Intervention Impact: Deletion vs. Confusion (SVN Agreement)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11, fontweight='bold')
    ax.axhline(0, color='black', linewidth=1)
    ax.legend(fontsize=11)
    
    # Value annotations
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                    f'{height:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    out_file = output_dir / "aggregated_results.png"
    plt.savefig(out_file, dpi=300)
    print(f"✓ Saved aggregated visualization to {out_file}")

def main():
    print("Aggregating SVN Circuit Analysis Results...\n" + "="*50)
    base_dir = Path(__file__).parent
    
    files_to_load = [
        ("Zero Ablation", base_dir / "zero_ablation" / "zero_ablation_results.json"),
        ("Mean Ablation", base_dir / "mean_ablation" / "mean_ablation_results.json"),
        ("Resample Ablation", base_dir / "resample_ablation" / "resample_ablation_results.json"),
        ("ACDC Pruning", base_dir / "acdc_svn_results" / "acdc_results.json")
    ]
    
    stats_list = []
    for name, filepath in files_to_load:
        records = load_results(filepath)
        stats = compute_stats(name, records)
        if stats:
            stats_list.append(stats)
            
    if stats_list:
        generate_markdown_table(stats_list)
        create_visualization(stats_list, base_dir)

    # Add causal scrubbing summary
    cs_filepath = base_dir / "causal_scrubbing_results.json"
    cs_data = load_causal_scrubbing(cs_filepath)
    if cs_data:
        report_causal_scrubbing(cs_data)

if __name__ == "__main__":
    main()