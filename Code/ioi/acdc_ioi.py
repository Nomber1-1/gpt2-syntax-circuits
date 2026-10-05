"""
Automated Circuit Discovery (ACDC) for Indirect Object Identification (IOI)
=====================================================================
This script implements a node-based iterative pruning approach inspired by ACDC.
It systematically prunes attention heads by replacing their outputs with 
corrupted activations (from corrupted prompts), keeping only the minimal 
circuit of heads necessary to maintain the baseline performance.
"""

import torch
import numpy as np
from functools import partial
from typing import List, Tuple, Dict, Set, Optional
import json
from pathlib import Path
from tqdm import tqdm
import sys

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)
    
from core.base_analyzer import BaseAblationAnalyzer
from core.ioi_dataset import get_ioi_test_cases, IOITestCase
from core.reporting import (
    save_ablation_results,
    create_ablation_visualizations,
    create_html_report_generic,
    generate_ablation_report_stats,
)

class ACDCIOIAnalyzer(BaseAblationAnalyzer):
    """Implements node-based ACDC for identifying the IOI circuit."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        self.pruned_heads: Set[Tuple[int, int]] = set()

    def patch_corrupted_z(self, z: torch.Tensor, hook, layer_idx: int, corrupted_cache):
        """Hook to patch in corrupted activations for pruned heads."""
        for head in range(self.num_heads):
            if (layer_idx, head) in self.pruned_heads:
                # Sequence lengths can differ between clean and corrupted prompts due to tokenization.
                # We right-align to ensure the final predictive tokens match.
                clean_seq_len = z.shape[1]
                corr_seq_len = corrupted_cache[hook.name].shape[1]
                min_len = min(clean_seq_len, corr_seq_len)
                
                # Replace the head's output with the activation from the corrupted prompt
                z[:, -min_len:, head, :] = corrupted_cache[hook.name][:, -min_len:, head, :]
        return z

    def run_acdc_discovery(self, threshold: float = 0.3) -> List[Tuple[int, int]]:
        """
        Runs dataset-wide node-based ACDC.
        
        Args:
            threshold: The maximum allowed average drop in logit diff. 
                       If dropping a head reduces performance by more than this, it is kept.
        """
        print("="*80)
        print(f"RUNNING NODE-BASED ACDC (Threshold = {threshold})")
        print("="*80)
        
        test_cases = get_ioi_test_cases()
        valid_cases = []
        corrupted_caches = {}
        clean_diffs = {}
        
        print("1. Precomputing baselines and corrupted caches...")
        for tc in tqdm(test_cases, desc="Precomputing baselines"):
            baseline = self.get_baseline_performance(tc)
            if baseline and baseline["is_valid"]:
                correct_id = baseline["correct_id"]
                incorrect_id = baseline["incorrect_id"]
                
                valid_cases.append((tc, correct_id, incorrect_id))
                
                # Baseline clean run
                clean_logits, _ = self.model.run_with_cache(tc.clean_prompt)
                clean_diff = self.evaluate_logit_difference(clean_logits, correct_id, incorrect_id)
                
                # Baseline corrupted run (for caching)
                _, cache = self.model.run_with_cache(tc.corrupted_prompt, remove_batch_dim=False)
                
                clean_diffs[tc.clean_prompt] = clean_diff
                corrupted_caches[tc.clean_prompt] = cache
        
        print(f"  Using {len(valid_cases)} valid test cases.\n")
        avg_baseline_diff = sum(clean_diffs.values()) / len(clean_diffs)
        
        # Initialize Circuit
        self.pruned_heads = set()
        circuit_heads = set([(l, h) for l in range(self.num_layers) for h in range(self.num_heads)])
        self.results = []
        
        print("2. Starting iterative pruning (Backwards from output to input)...")
        # Iterate backwards because late layers write directly to the residual stream.
        
        # Pre-calculate hook names to avoid string formatting in loops
        hook_names = [f"blocks.{l}.attn.hook_z" for l in range(self.num_layers)]
        
        all_heads_to_prune = [(l, h) for l in range(self.num_layers - 1, -1, -1) for h in range(self.num_heads)]

        for layer, head in tqdm(all_heads_to_prune, desc="ACDC Pruning"):
                # Propose pruning this head
                self.pruned_heads.add((layer, head))
                circuit_heads.remove((layer, head))
                
                total_drop = 0.0
                
                # Register hooks once for this pruning configuration
                hooks = []
                for l in range(self.num_layers):
                    hooks.append((hook_names[l], partial(self.patch_corrupted_z, layer_idx=l, corrupted_cache=None))) # Cache passed via tc
                
                # Evaluate performance on all valid test cases with this new pruning configuration
                for tc, correct_id, incorrect_id in valid_cases:
                    # Note: patch_corrupted_z uses self.model.run_with_hooks which handles the cache via the partial's logic
                    # but we need to pass the specific cache for THIS test case. 
                    # Optimization: We update the partial's keyword argument dynamically or use a more flexible hook.
                    
                    # To keep it clean and working with your current hook signature:
                    current_hooks = [(name, partial(func.func, layer_idx=func.keywords['layer_idx'], 
                                                 corrupted_cache=corrupted_caches[tc.clean_prompt])) 
                                    for name, func in hooks]
                    
                    ablated_logits = self.model.run_with_hooks(tc.clean_prompt, fwd_hooks=current_hooks)
                    ablated_diff = self.evaluate_logit_difference(ablated_logits, correct_id, incorrect_id)
                    
                    # Calculate drop
                    drop = clean_diffs[tc.clean_prompt] - ablated_diff
                    total_drop += drop
                
                avg_drop = total_drop / len(valid_cases)
                
                # Decision: Keep or Prune
                if avg_drop > threshold:
                    # Revert the prune: Head is important!
                    self.pruned_heads.remove((layer, head))
                    circuit_heads.add((layer, head))
                    print(f"  [Keep] Layer {layer:2d}, Head {head:2d} (Drop: {avg_drop:6.4f} > {threshold})")
                    
                # Record the evaluation for reporting and visualization
                self.results.append({
                    "layer": layer,
                    "head": head,
                    "performance_drop": avg_drop,
                    "baseline_diff": avg_baseline_diff,
                    "ablated_diff": avg_baseline_diff - avg_drop,
                    "is_in_circuit": avg_drop > threshold,
                    "case_type": "dataset_average"
                })
                    
        print("\n" + "="*80)
        print(f"DISCOVERED CIRCUIT ({len(circuit_heads)}/{self.num_layers * self.num_heads} heads)")
        print("="*80)
        
        sorted_circuit = sorted(list(circuit_heads))
        for l, h in sorted_circuit:
            print(f"  Layer {l}, Head {h}")
            
        # Sort results sequentially for the reporting tools
        self.results.sort(key=lambda x: (x["layer"], x["head"]))
        return sorted_circuit

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("ACDC DISCOVERY REPORT")
        print("="*80)
        
        if not hasattr(self, 'results') or not self.results:
            print("No results to report")
            return {}
            
        stats = generate_ablation_report_stats(self.results)
        
        print(f"  Min Drop: {stats['min_drop']:.4f}")
        print(f"  Max Drop: {stats['max_drop']:.4f}")
        
        circuit_size = len([r for r in self.results if r.get("is_in_circuit")])
        print(f"\nFinal circuit contains {circuit_size} heads.")
            
        return stats
    
    def save_results(self, filename: str = "acdc_results.json"):
        """Save detailed ACDC results to JSON."""
        output_path = Path(__file__).parent / "acdc_ioi_results" / filename
        output_path.parent.mkdir(exist_ok=True)
        combined_results = {
            "valid_ablations": self.results,
            "circuit_size": len([r for r in self.results if r.get("is_in_circuit")])
        }
        save_ablation_results(combined_results, output_path)

    def create_visualizations(self):
        """Create a heatmap of the performance drops during ACDC pruning."""
        if not hasattr(self, 'results') or not self.results:
            print("No results to visualize")
            return
        
        output_path = Path(__file__).parent / "acdc_ioi_results" / "acdc_visualization.png"
        output_path.parent.mkdir(exist_ok=True)
        viz_path = create_ablation_visualizations(
            self.results, 
            self.num_layers, 
            self.num_heads, 
            "ACDC Pruning Drops", 
            output_path
        )
        print(f"Visualization saved to {viz_path}")
        return viz_path

    def create_html_report(self, summary: Dict):
        """Create an interactive HTML report."""
        viz_path = self.create_visualizations()
        output_path = Path(__file__).parent / "acdc_ioi_results" / "acdc_report.html"
        output_path.parent.mkdir(exist_ok=True)
        
        description = (
            "<strong>ACDC Discovery:</strong> This analysis applies automated circuit discovery using iterative node pruning. "
            "Heads are evaluated backwards; if their removal drops performance by more than the threshold, they are kept in the circuit. "
            "The heatmap below shows the average exact-match logit drop evaluated at the moment each head was tested for pruning."
        )
        
        create_html_report_generic(
            results=self.results,
            summary=summary,
            viz_path=viz_path,
            output_path=output_path,
            title="ACDC Circuit Discovery Report",
            description_html=description
        )

if __name__ == "__main__":
    analyzer = ACDCIOIAnalyzer()
    # Adjust threshold to change circuit sparsity. Lower = larger circuit, Higher = sparser circuit.
    circuit = analyzer.run_acdc_discovery(threshold=0.3)
    summary = analyzer.generate_report()
    if summary:
        analyzer.save_results()
        analyzer.create_html_report(summary)