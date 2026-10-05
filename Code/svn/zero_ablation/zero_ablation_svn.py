"""
Zero Ablation Script for Subject-Verb Number Agreement
========================================================
This script implements complete Zero Ablation analysis for identifying which 
attention heads are critical for subject-verb number agreement in GPT-2 Small.

Zero Ablation: Sets a component's output vector to zero, measuring the resulting
performance degradation to understand the component's importance.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from functools import partial
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm

from core.base_analyzer import BaseAblationAnalyzer
from core.reporting import (
    save_ablation_results,
    create_ablation_visualizations,
    create_html_report_generic,
    generate_ablation_report_stats,
)
from core.svn_dataset import get_svn_test_cases, get_svn_dataset_prompts, SVNTestCase

class ZeroAblationAnalyzer(BaseAblationAnalyzer):
    """Implements complete Zero Ablation analysis for SVN agreement."""
    
    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        
    def create_test_cases(self, limit: Optional[int] = None) -> List[SVNTestCase]:
        """Return standardized SVN test cases from the shared dataset."""
        return get_svn_test_cases(limit=limit)
    
    def zero_ablate_head(self, z: torch.Tensor, hook, head_index: int) -> torch.Tensor:
        """Hook function to zero out a specific attention head."""
        # z shape: [batch, position, n_heads, d_head]
        z[:, :, head_index, :] = 0.0
        return z
    
    def run_zero_ablation_on_case(self, test_case: SVNTestCase, 
                                  baseline: Dict) -> Optional[List[Dict]]:
        """Run zero ablation for all heads on a single test case."""
        if baseline is None or not baseline["is_valid"]:
            return None
        
        ablation_results = []
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                try:
                    hook_name = f"blocks.{layer}.attn.hook_z"
                    
                    # Run model with zero-ablation hook on clean prompt
                    ablated_logits_clean = self.model.run_with_hooks(
                        test_case.clean_prompt,
                        fwd_hooks=[(hook_name, partial(self.zero_ablate_head, head_index=head))]
                    )
                    ablated_clean_diff = self.evaluate_logit_difference(
                        ablated_logits_clean, baseline["correct_id"], baseline["incorrect_id"]
                    )
                    
                    # Run model with zero-ablation hook on corrupted prompt
                    ablated_logits_corrupted = self.model.run_with_hooks(
                        test_case.corrupted_prompt,
                        fwd_hooks=[(hook_name, partial(self.zero_ablate_head, head_index=head))]
                    )
                    ablated_corrupted_diff = self.evaluate_logit_difference(
                        ablated_logits_corrupted, baseline["correct_id"], baseline["incorrect_id"]
                    )
                    
                    ablated_delta = ablated_clean_diff - ablated_corrupted_diff
                    
                    # Calculate performance drop using Delta Delta
                    performance_drop = baseline["base_diff_delta"] - ablated_delta
                    
                    ablation_results.append({
                        "layer": layer,
                        "head": head,
                        "baseline_diff": baseline["base_diff_delta"],
                        "ablated_diff": ablated_delta,
                        "performance_drop": performance_drop,
                        "case_type": test_case.case_type
                    })
                except Exception as e:
                    print(f"Error ablating layer {layer}, head {head}: {e}")
                    continue
        
        return ablation_results
    


    def run_complete_analysis(self, limit: Optional[int] = None):
        """Run complete zero ablation analysis across all test cases."""
        print("="*80)
        print("ZERO ABLATION ANALYSIS FOR SUBJECT-VERB NUMBER AGREEMENT")
        print("="*80)
        
        test_cases = self.create_test_cases(limit=limit)
        print(f"\nRunning zero ablation on {len(test_cases)} test cases...")
        print(f"Model: {self.num_layers} layers × {self.num_heads} heads = {self.num_layers * self.num_heads} total heads\n")
        
        all_ablation_data = []
        self.failed_baselines = []
        valid_cases = 0
        
        with torch.no_grad():
            for test_case in tqdm(test_cases, desc="Running Zero Ablation"):
                # Get baseline
                baseline = self.get_baseline_performance(test_case)
                if baseline is None or not baseline["is_valid"]:
                    self.failed_baselines.append({
                        "prompt": test_case.clean_prompt,
                        "case_type": test_case.case_type,
                        "reason": "baseline mismatch",
                        "baseline_data": baseline
                    })
                    continue
                
                # Run zero ablation
                ablation_results = self.run_zero_ablation_on_case(test_case, baseline)
                if ablation_results:
                    all_ablation_data.extend(ablation_results)
                    valid_cases += 1
        
        self.results = all_ablation_data
        print(f"\n✓ Completed analysis on {valid_cases} valid test cases")
        print(f"Total ablation records: {len(self.results)}\n")

    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("ZERO ABLATION ANALYSIS REPORT")
        print("="*80)
        
        if not self.results:
            print("No ablation results to report")
            return {}
            
        stats = generate_ablation_report_stats(self.results)
        
        print(f"  Min Drop (least impact): {stats['min_drop']:.4f}")
        print(f"  Max Drop (most impact):  {stats['max_drop']:.4f}")
        print(f"  Median Drop:            {stats['median_drop']:.4f}")
        
        print(f"\nTop 10 Most Important Heads (Highest Performance Drop):")
        for idx, (layer, head, drop) in enumerate(stats['top_heads'], 1):
            print(f"  {idx:2d}. Layer {layer:2d}, Head {head:2d} | Drop: {drop:7.4f}")
            
        return stats
    
    def save_results(self, filename: str = "zero_ablation_results.json"):
        """Save detailed ablation results to JSON."""
        output_path = __import__("pathlib").Path(__file__).parent / filename
        
        combined_results = {
            "valid_ablations": self.results,
            "failed_baselines": self.failed_baselines
        }
        save_ablation_results(combined_results, output_path)
    
    def create_visualizations(self):
        """Create visualizations of ablation results."""
        if not self.results:
            print("No results to visualize")
            return
            
        output_path = __import__("pathlib").Path(__file__).parent / "zero_ablation_visualization.png"
        viz_path = create_ablation_visualizations(
            self.results, 
            self.num_layers, 
            self.num_heads, 
            "Zero Ablation", 
            output_path
        )
        print(f"Visualization saved to {viz_path}")
        return viz_path
    
    def create_html_report(self, summary: Dict):
        """Create an interactive HTML report."""
        viz_path = self.create_visualizations()
        output_path = __import__("pathlib").Path(__file__).parent / "zero_ablation_report.html"
        
        description = (
            "<strong>Zero Ablation Method:</strong> This analysis applies zero ablation to identify critical attention heads\n"
            "for subject-verb number agreement. Zero ablation replaces a component's output with a zero vector. This is the\n"
            "simplest form of ablation and serves as a baseline for more sophisticated methods, though it can sometimes\n"
            "introduce out-of-distribution behavior."
        )
        
        create_html_report_generic(
            results=self.results,
            summary=summary,
            viz_path=viz_path,
            output_path=output_path,
            title="Zero Ablation Analysis Report",
            description_html=description
        )


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Zero Ablation analysis for SVN.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test cases for a faster run.")
    args = parser.parse_args()

    analyzer = ZeroAblationAnalyzer()
    analyzer.run_complete_analysis(limit=args.limit)
    summary = analyzer.generate_report()
    if summary:
        analyzer.save_results()
        analyzer.create_html_report(summary)
        
    print("\n" + "="*80)
    print("✓ Zero Ablation analysis complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
