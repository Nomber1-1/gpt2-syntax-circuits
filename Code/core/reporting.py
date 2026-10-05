"""
Shared Reporting Utilities
==========================
"""
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

def save_ablation_results(results: List[Dict], filepath: Path):
    """Save detailed ablation results to JSON."""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Detailed results saved to {filepath}")

def generate_ablation_report_stats(results: List[Dict], drop_key: str = "performance_drop"):
    """Generate common statistics dict for ablation results."""
    all_drops = [r[drop_key] for r in results]
    results_sorted = sorted(results, key=lambda x: x[drop_key], reverse=True)
    
    case_types = {}
    for result in results:
        case = result["case_type"]
        if case not in case_types:
            case_types[case] = []
        case_types[case].append(result[drop_key])
        
    return {
        "total_ablations": len(results),
        "mean_drop": float(np.mean(all_drops)),
        "std_drop": float(np.std(all_drops)),
        "max_drop": float(np.max(all_drops)),
        "min_drop": float(np.min(all_drops)),
        "median_drop": float(np.median(all_drops)),
        "top_heads": [(r["layer"], r["head"], r[drop_key]) for r in results_sorted[:10]],
        "case_types": {k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "median": float(np.median(v))} 
                       for k, v in case_types.items()}
    }

def create_ablation_visualizations(results: List[Dict], num_layers: int, num_heads: int, 
                                   method_name: str, output_path: Path, drop_key: str = "performance_drop"):
    """Create standard comprehensive visualizations of ablation results."""
    if not results:
        print("No results to visualize")
        return

    # Set style
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=(18, 12))
    
    # 1. Heatmap of performance drops by layer and head (for first case type)
    ax1 = plt.subplot(2, 3, 1)
    
    # Get data for first case type
    case_type = results[0].get("case_type", "N/A")
    filtered_results = [r for r in results if r.get("case_type") == case_type]
    
    # Create matrix
    drop_matrix = np.zeros((num_layers, num_heads))
    for result in filtered_results:
        drop_matrix[result["layer"], result["head"]] = result[drop_key]
    
    im = ax1.imshow(drop_matrix, cmap="RdYlGn_r", aspect="auto")
    ax1.set_xlabel("Head Index", fontweight="bold")
    ax1.set_ylabel("Layer", fontweight="bold")
    ax1.set_title(f"Performance Drop Heatmap\n({case_type})", fontweight="bold")
    plt.colorbar(im, ax=ax1, label="Performance Drop")
    
    # 2. Distribution of performance drops
    ax2 = plt.subplot(2, 3, 2)
    all_drops = [r[drop_key] for r in results]
    ax2.hist(all_drops, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
    mean_drop = np.mean(all_drops)
    ax2.axvline(mean_drop, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_drop:.4f}")
    ax2.set_xlabel("Performance Drop", fontweight="bold")
    ax2.set_ylabel("Frequency", fontweight="bold")
    ax2.set_title(f"Distribution of Drops ({method_name})", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Top 15 heads by performance drop
    ax3 = plt.subplot(2, 3, 3)
    results_sorted = sorted(results, key=lambda x: x[drop_key], reverse=True)[:15]
    head_labels = [f"L{r['layer']}H{r['head']}" for r in results_sorted]
    drops = [r[drop_key] for r in results_sorted]
    colors = plt.cm.Reds(np.linspace(0.4, 1, len(drops)))
    ax3.barh(head_labels, drops, color=colors, edgecolor="black")
    ax3.set_xlabel("Performance Drop", fontweight="bold")
    ax3.set_title("Top 15 Most Important Heads", fontweight="bold")
    ax3.invert_yaxis()
    
    # 4. Average drop by layer
    ax4 = plt.subplot(2, 3, 4)
    layer_drops = {}
    for result in results:
        layer = result["layer"]
        if layer not in layer_drops:
            layer_drops[layer] = []
        layer_drops[layer].append(result[drop_key])
    
    layer_means = [np.mean(layer_drops[l]) for l in sorted(layer_drops.keys())]
    ax4.plot(sorted(layer_drops.keys()), layer_means, marker="o", linewidth=2, markersize=8, color="#e74c3c")
    ax4.set_xlabel("Layer", fontweight="bold")
    ax4.set_ylabel("Average Performance Drop", fontweight="bold")
    ax4.set_title("Average Drop by Layer", fontweight="bold")
    ax4.grid(alpha=0.3)
    
    # 5. Drops by case type
    ax5 = plt.subplot(2, 3, 5)
    case_types = {}
    for result in results:
        case = result.get("case_type", "Unknown")
        if case not in case_types:
            case_types[case] = []
        case_types[case].append(result[drop_key])
    
    case_names = sorted(case_types.keys())
    case_means = [np.mean(case_types[c]) for c in case_names]
    case_maxs = [np.max(case_types[c]) for c in case_names]
    
    x_pos = np.arange(len(case_names))
    ax5.bar(x_pos, case_means, color="#2ecc71", label="Mean", alpha=0.7, edgecolor="black")
    ax5.bar(x_pos, case_maxs, color="#e74c3c", label="Max", alpha=0.5, edgecolor="black")
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(case_names, rotation=45, ha="right")
    ax5.set_ylabel("Performance Drop", fontweight="bold")
    ax5.set_title("Drop Statistics by Case Type", fontweight="bold")
    ax5.legend()
    ax5.grid(alpha=0.3, axis="y")
    
    # 6. Scatter: baseline diff vs performance drop
    ax6 = plt.subplot(2, 3, 6)
    if all("baseline_diff" in r for r in results):
        baseline_diffs = [r["baseline_diff"] for r in results]
        perf_drops = [r[drop_key] for r in results]
        scatter = ax6.scatter(baseline_diffs, perf_drops, alpha=0.6, c=perf_drops, cmap="plasma", edgecolors="black")
        ax6.set_xlabel("Baseline Logit Difference", fontweight="bold")
        ax6.set_ylabel("Performance Drop", fontweight="bold")
        ax6.set_title("Baseline vs Performance Drop", fontweight="bold")
        plt.colorbar(scatter, ax=ax6, label="Drop")
    else:
        ax6.text(0.5, 0.5, "Baseline data missing", ha="center", va="center")
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    return output_path


import base64
from datetime import datetime



def create_html_report_generic(results, summary, viz_path, output_path, title, description_html):
    import base64
    img_data = ""
    if viz_path:
        with open(viz_path, 'rb') as vf:
            img_data = base64.b64encode(vf.read()).decode()
            
    results_sorted = sorted(results, key=lambda x: x.get('performance_drop', 0), reverse=True)
    top_heads_html = ""
    for idx, result in enumerate(results_sorted[:15], 1):
        top_heads_html += f"""
            <tr>
                <td>{idx}</td>
                <td>Layer {result['layer']}</td>
                <td>Head {result['head']}</td>
                <td>{result.get('performance_drop', 0.0):.4f}</td>
                <td>{result.get('ablated_diff', 0.0):.4f}</td>
                <td>{result.get('case_type', 'N/A')}</td>
            </tr>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: Segoe UI, Tahoma, sans-serif;
            margin: 0;
            padding: 24px;
            background: linear-gradient(135deg, #f7f7fb 0%, #eef3ff 100%);
            color: #1f2937;
        }}
        .card {{
            max-width: 1280px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 18px 50px rgba(15, 23, 42, 0.12);
            overflow: hidden;
        }}
        .header {{
            padding: 32px 36px;
            background: linear-gradient(135deg, #1f2937 0%, #334155 100%);
            color: white;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 30px; }}
        .header p {{ margin: 0; opacity: 0.9; }}
        .content {{ padding: 32px 36px 40px 36px; }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 28px;
        }}
        .stat-card {{
            padding: 16px;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            background: #f8fafc;
        }}
        .stat-card .label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: #6b7280; }}
        .stat-card .value {{ font-size: 24px; font-weight: 700; margin-top: 8px; }}
        .description {{
            background: #f1f5f9;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 4px;
            line-height: 1.6;
        }}
        .figure {{ text-align: center; margin: 18px 0 30px; }}
        .figure img {{ max-width: 100%; height: auto; border-radius: 12px; border: 1px solid #e5e7eb; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 12px;
            overflow: hidden;
            border-radius: 12px;
        }}
        th {{
            text-align: left;
            background: #334155;
            color: white;
            padding: 12px 14px;
            font-size: 13px;
        }}
        td {{ padding: 12px 14px; border-bottom: 1px solid #e5e7eb; }}
        tr:nth-child(even) td {{ background: #f8fafc; }}
        .footer {{
            margin-top: 20px;
            color: #6b7280;
            font-size: 13px;
            text-align: center;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="card">
        <div class="header">
            <h1>{title}</h1>
            <p>Mechanistic Interpretability Analysis Report</p>
        </div>
        <div class="content">
            <div class="description">{description_html}</div>
            <div class="summary-grid">
                <div class="stat-card"><div class="label">Total Heads</div><div class="value">{summary.get('total_ablations', len(results))}</div></div>
                <div class="stat-card"><div class="label">Mean Drop</div><div class="value">{summary.get('mean_drop', 0.0):.4f}</div></div>
                <div class="stat-card"><div class="label">Max Drop</div><div class="value">{summary.get('max_drop', 0.0):.4f}</div></div>
                <div class="stat-card"><div class="label">Median Drop</div><div class="value">{summary.get('median_drop', 0.0):.4f}</div></div>
            </div>
            {f'<div class="figure"><img src="data:image/png;base64,{img_data}" alt="Ablation Visualization" /></div>' if img_data else ''}
            <h2>Top 15 Most Important Heads</h2>
            <table>
                <thead><tr><th>Rank</th><th>Layer</th><th>Head</th><th>Drop</th><th>Ablated Logit Diff</th><th>Case Type</th></tr></thead>
                <tbody>{top_heads_html}</tbody>
            </table>
        </div>
        <div class="footer">Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write(html)
