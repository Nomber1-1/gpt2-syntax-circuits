"""
Mean Ablation Script for Subject-Verb Number Agreement
=======================================================
This script implements complete Mean Ablation analysis for identifying which 
attention heads are critical for subject-verb number agreement in GPT-2 Small.

Mean Ablation: Replaces a component's output with its average activation
calculated across an entire dataset. This maintains normal activation levels
while stripping away any information related to the current prompt.
"""

import torch
import numpy as np
from transformer_lens import HookedTransformer
from functools import partial
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime

from core.base_analyzer import BaseAblationAnalyzer
from core.reporting import save_ablation_results
from core.svn_dataset import get_svn_test_cases, get_svn_dataset_prompts, SVNTestCase


class MeanAblationAnalyzer(BaseAblationAnalyzer):
    """Implements complete Mean Ablation analysis for SVN agreement."""

    def __init__(self, device: Optional[str] = None):
        super().__init__(device=device)
        self.mean_activations = {}  # Will store mean activations per head
        
    def create_dataset_prompts(self) -> List[str]:
        """Delegate dataset prompts to the shared dataset."""
        return get_svn_dataset_prompts()
    
    def create_test_cases(self, limit: Optional[int] = None) -> List[SVNTestCase]:
        """Return standardized SVN test cases from the shared dataset."""
        return get_svn_test_cases(limit=limit)
    
    def compute_mean_activations(self):
        """Compute mean activations across the dataset for each head."""
        print("\n" + "="*80)
        print("COMPUTING MEAN ACTIVATIONS FROM DATASET")
        print("="*80)
        
        dataset_prompts = self.create_dataset_prompts()
        print(f"\nDataset size: {len(dataset_prompts)} prompts")
        
        # Initialize storage for activations
        activation_sums = {}
        for layer in range(self.num_layers):
            activation_sums[layer] = {}
            for head in range(self.num_heads):
                activation_sums[layer][head] = None
        
        print("\nCollecting activations from dataset...")
        
        # Collect activations from all prompts
        with torch.no_grad():
            for prompt in tqdm(dataset_prompts, desc="Collecting activations"):
                try:
                    # Create hook to capture z activations (before attention softmax)
                    z_activations = {}
                    
                    def capture_z(z, hook, layer_idx):
                        z_activations[layer_idx] = z.detach().clone()
                        return z
                    
                    # Register hooks for all layers
                    hooks = []
                    for layer in range(self.num_layers):
                        hook_name = f"blocks.{layer}.attn.hook_z"
                        hooks.append((hook_name, partial(capture_z, layer_idx=layer)))
                    
                    # Run model with hooks
                    _, cache = self.model.run_with_cache(prompt, remove_batch_dim=False)
                    
                    # Extract z activations from cache instead
                    for layer in range(self.num_layers):
                        hook_name = f"blocks.{layer}.attn.hook_z"
                        if hook_name in cache:
                            z = cache[hook_name]  # Shape: [batch=1, position, n_heads, d_head]
                            
                            # Average over position dimension
                            z_mean = z.mean(dim=1)  # Shape: [1, n_heads, d_head]
                            
                            for head in range(self.num_heads):
                                head_activation = z_mean[0, head, :]  # Shape: [d_head]
                                
                                if activation_sums[layer][head] is None:
                                    activation_sums[layer][head] = head_activation.cpu()
                                else:
                                    activation_sums[layer][head] += head_activation.cpu()
                    
                except Exception as e:
                    tqdm.write(f"  Warning: Error processing prompt: {e}")
                    continue
        
        # Compute means
        print("\nComputing means...")
        for layer in range(self.num_layers):
            self.mean_activations[layer] = {}
            for head in range(self.num_heads):
                if activation_sums[layer][head] is not None:
                    self.mean_activations[layer][head] = activation_sums[layer][head] / len(dataset_prompts)
                    self.mean_activations[layer][head] = self.mean_activations[layer][head].to(self.device)
                else:
                    # Fallback to zero if no activations collected
                    self.mean_activations[layer][head] = torch.zeros(self.d_head, device=self.device)
        
        print("✓ Mean activations computed for all heads\n")
    
    def mean_ablate_head(self, z: torch.Tensor, hook, layer_idx: int, head_index: int) -> torch.Tensor:
        """Hook function to replace a specific head with mean activation."""
        if layer_idx in self.mean_activations and head_index in self.mean_activations[layer_idx]:
            mean_activation = self.mean_activations[layer_idx][head_index]
            # z shape: [batch, position, n_heads, d_head]
            z[:, :, head_index, :] = mean_activation.unsqueeze(0).unsqueeze(0)
        return z
    
    def run_mean_ablation_on_case(self, test_case: SVNTestCase, 
                                  baseline: Dict) -> Optional[List[Dict]]:
        """Run mean ablation for all heads on a single test case."""
        if baseline is None or not baseline["is_valid"]:
            return None
        
        ablation_results = []
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                try:
                    hook_name = f"blocks.{layer}.attn.hook_z"
                    
                    # Run model with mean-ablation hook on clean prompt
                    ablated_logits_clean = self.model.run_with_hooks(
                        test_case.clean_prompt,
                        fwd_hooks=[(hook_name, partial(self.mean_ablate_head, layer_idx=layer, head_index=head))]
                    )
                    ablated_clean_diff = self.evaluate_logit_difference(
                        ablated_logits_clean, baseline["correct_id"], baseline["incorrect_id"]
                    )
                    
                    # Run model with mean-ablation hook on corrupted prompt
                    ablated_logits_corrupted = self.model.run_with_hooks(
                        test_case.corrupted_prompt,
                        fwd_hooks=[(hook_name, partial(self.mean_ablate_head, layer_idx=layer, head_index=head))]
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
        """Run complete mean ablation analysis across all test cases."""
        # First compute mean activations
        self.compute_mean_activations()
        
        print("="*80)
        print("MEAN ABLATION ANALYSIS FOR SUBJECT-VERB NUMBER AGREEMENT")
        print("="*80)
        
        test_cases = self.create_test_cases(limit=limit)
        print(f"\nRunning mean ablation on {len(test_cases)} test cases...")
        print(f"Model: {self.num_layers} layers × {self.num_heads} heads = {self.num_layers * self.num_heads} total heads\n")
        
        all_ablation_data = []
        self.failed_baselines = []
        valid_cases = 0
        
        with torch.no_grad():
            for test_case in tqdm(test_cases, desc="Running Mean Ablation"):
                # Get baseline
                baseline = self.get_baseline_performance(test_case)
                if baseline is None or not baseline["is_valid"]:
                    self.failed_baselines.append({
                        "prompt": test_case.clean_prompt,
                        "case_type": test_case.case_type,
                        "reason": "baseline mismatch or error",
                        "baseline_data": baseline
                    })
                    continue
                
                # Run mean ablation
                ablation_results = self.run_mean_ablation_on_case(test_case, baseline)
                if ablation_results:
                    all_ablation_data.extend(ablation_results)
                    valid_cases += 1
        
        self.results = all_ablation_data
        print(f"\n✓ Completed analysis on {valid_cases} valid test cases")
        print(f"Failed / invalid baseline test cases: {len(self.failed_baselines)}")
        print(f"Total ablation records: {len(self.results)}\n")
    
    def generate_report(self) -> Dict:
        """Generate comprehensive analysis report."""
        print("\n" + "="*80)
        print("MEAN ABLATION ANALYSIS REPORT")
        print("="*80)
        
        if not self.results:
            print("No ablation results to report")
            return {}
            
        from core.reporting import generate_ablation_report_stats
        stats = generate_ablation_report_stats(self.results)
        
        print(f"  Min Drop (least impact): {stats['min_drop']:.4f}")
        print(f"  Max Drop (most impact):  {stats['max_drop']:.4f}")
        print(f"  Median Drop:            {stats['median_drop']:.4f}")
        
        print(f"\nTop 10 Most Important Heads (Highest Performance Drop):")
        for idx, (layer, head, drop) in enumerate(stats['top_heads'], 1):
            print(f"  {idx:2d}. Layer {layer:2d}, Head {head:2d} | Drop: {drop:7.4f}")
            
        return stats
    
    def save_results(self, filename: str = "mean_ablation_results.json"):
        """Save detailed ablation results to JSON."""
        output_path = __import__("pathlib").Path(__file__).parent / filename
        
        # We can also save out the failed baselines so we don't lose that data
        combined_results = {
            "valid_ablations": self.results,
            "failed_baselines": self.failed_baselines
        }
        
        save_ablation_results(combined_results, output_path)
    
    def create_visualizations(self):
        """Create comprehensive visualizations of ablation results."""
        if not self.results:
            print("No results to visualize")
            return
            
        from core.reporting import create_ablation_visualizations
        output_path = __import__("pathlib").Path(__file__).parent / "mean_ablation_visualization.png"
        
        viz_path = create_ablation_visualizations(
            self.results, 
            self.num_layers, 
            self.num_heads, 
            "Mean Ablation", 
            output_path
        )
        print(f"Visualization saved to {viz_path}")
        return viz_path

    def create_html_report(self, summary: Dict):
        """Create an interactive HTML report."""
        from core.reporting import create_html_report_generic
        output_path = __import__("pathlib").Path(__file__).parent / "mean_ablation_report.html"
        viz_path = self.create_visualizations()
        
        description = (
            "<strong>Mean Ablation Method:</strong> This analysis applies mean ablation to identify critical attention heads\n"
            "for subject-verb number agreement. Mean ablation replaces a component's output with its average activation\n"
            "calculated across the entire dataset. This maintains normal activation levels while stripping away prompt-specific\n"
            "information. This method is more robust than zero ablation as it avoids triggering unpredictable failures."
        )
        
        create_html_report_generic(
            results=self.results,
            summary=summary,
            viz_path=viz_path,
            output_path=output_path,
            title="Mean Ablation Analysis Report",
            description_html=description
        )

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Mean Ablation analysis for SVN.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test cases for a faster run.")
    args = parser.parse_args()

    analyzer = MeanAblationAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis(limit=args.limit)
    
    # Generate report
    summary = analyzer.generate_report()
    if summary:
        # Save results
        analyzer.save_results()
        
        # Create visualizations and HTML report
        print("Generating visualizations and HTML report...")
        analyzer.create_html_report(summary)
        
    print("\n" + "="*80)
    print("✓ Mean Ablation analysis complete!")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
