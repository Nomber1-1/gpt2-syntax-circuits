"""
Resample Ablation Script for Subject-Verb Number Agreement
===========================================================
This script implements complete Resample Ablation analysis for identifying which 
attention heads are critical for subject-verb number agreement in GPT-2 Small.

Resample Ablation: Replaces a component's output with an activation obtained 
from a randomly sampled prompt. This ensures the model receives data that is 
both normal (proper activation distribution) and independent of context.
"""

import torch
import numpy as np
from functools import partial
from typing import List, Tuple, Dict, Optional
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import random
from tqdm import tqdm

from core.base_analyzer import BaseAblationAnalyzer
from core.reporting import save_ablation_results
from core.svn_dataset import get_svn_test_cases, get_svn_dataset_prompts, SVNTestCase


class ResampleAblationAnalyzer(BaseAblationAnalyzer):
    """Implements complete Resample Ablation analysis for SVN agreement."""
    
    def __init__(self, device: Optional[str] = None, seed: int = 42):
        """Initialize the analyzer and load the model."""
        super().__init__(device=device)
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.cached_activations = {}  # Will store activations from dataset for resampling
        
    def create_dataset_prompts(self) -> List[str]:
        """Delegate dataset prompts to the shared dataset."""
        return get_svn_dataset_prompts()
    
    def create_test_cases(self, limit: Optional[int] = None) -> List[SVNTestCase]:
        """Return standardized SVN test cases from the shared dataset."""
        return get_svn_test_cases(limit=limit)
    
    def collect_activations_for_resampling(self):
        """Collect activations from dataset for resampling pool."""
        print("\n" + "="*80)
        print("COLLECTING ACTIVATIONS FOR RESAMPLING")
        print("="*80)
        
        dataset_prompts = self.create_dataset_prompts()
        print(f"\nDataset size: {len(dataset_prompts)} prompts")
        print("Building activation pool for each layer and head...\n")
        
        # Initialize storage for activations
        for layer in range(self.num_layers):
            self.cached_activations[layer] = {}
            for head in range(self.num_heads):
                self.cached_activations[layer][head] = []
        
        # Collect activations from all prompts
        with torch.no_grad():
            for prompt in tqdm(dataset_prompts, desc="Collecting activations"):
                try:
                    # Run model and extract cache
                    _, cache = self.model.run_with_cache(prompt, remove_batch_dim=False)
                    
                    # Extract z activations from cache
                    for layer in range(self.num_layers):
                        hook_name = f"blocks.{layer}.attn.hook_z"
                        if hook_name in cache:
                            z = cache[hook_name]  # Shape: [batch=1, position, n_heads, d_head]
                            
                            # For each head, collect all position activations
                            for head in range(self.num_heads):
                                head_activation = z[0, :, head, :]  # Shape: [position, d_head]
                                # Store all position activations for this head from this prompt
                                self.cached_activations[layer][head].extend(
                                    head_activation.detach().cpu().numpy()
                                )
                    
                except Exception as e:
                    tqdm.write(f"  Warning: Error processing prompt: {e}")
                    continue
        
        # Convert to arrays for faster sampling
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                if self.cached_activations[layer][head]:
                    self.cached_activations[layer][head] = np.array(
                        self.cached_activations[layer][head]
                    )
        
        print("✓ Activation pool collected for all heads\n")
    
    def resample_ablate_head(self, z: torch.Tensor, hook, layer_idx: int, head_index: int) -> torch.Tensor:
        """Hook function to replace a specific head with randomly sampled activation."""
        if layer_idx in self.cached_activations and head_index in self.cached_activations[layer_idx]:
            activation_pool = self.cached_activations[layer_idx][head_index]
            if len(activation_pool) > 0:
                # Randomly sample an activation from the pool
                sampled_idx = np.random.randint(0, len(activation_pool))
                sampled_activation = activation_pool[sampled_idx]  # Shape: [d_head]
                
                # Replace all positions in this head with the sampled activation
                # z shape: [batch, position, n_heads, d_head]
                z[:, :, head_index, :] = torch.from_numpy(sampled_activation).float().to(self.device).unsqueeze(0).unsqueeze(0)
        return z
    
    def run_resample_ablation_on_case(self, test_case: SVNTestCase, 
                                      baseline: Dict, num_samples: int = 5) -> Optional[List[Dict]]:
        """Run resample ablation for all heads on a single test case with multiple samples."""
        if baseline is None or not baseline["is_valid"]:
            return None
        
        ablation_results = []
        
        for layer in range(self.num_layers):
            for head in range(self.num_heads):
                # Run multiple samples and average results
                sample_drops = []
                
                for sample_idx in range(num_samples):
                    try:
                        hook_name = f"blocks.{layer}.attn.hook_z"
                        
                        # Run model with resample-ablation hook on clean prompt
                        ablated_logits_clean = self.model.run_with_hooks(
                            test_case.clean_prompt,
                            fwd_hooks=[(hook_name, partial(self.resample_ablate_head, 
                                                           layer_idx=layer, head_index=head))]
                        )
                        ablated_clean_diff = self.evaluate_logit_difference(
                            ablated_logits_clean, baseline["correct_id"], baseline["incorrect_id"]
                        )
                        
                        # Run model with resample-ablation hook on corrupted prompt
                        ablated_logits_corrupted = self.model.run_with_hooks(
                            test_case.corrupted_prompt,
                            fwd_hooks=[(hook_name, partial(self.resample_ablate_head, 
                                                           layer_idx=layer, head_index=head))]
                        )
                        ablated_corrupted_diff = self.evaluate_logit_difference(
                            ablated_logits_corrupted, baseline["correct_id"], baseline["incorrect_id"]
                        )
                        
                        ablated_delta = ablated_clean_diff - ablated_corrupted_diff
                        
                        # Calculate performance drop using Delta Delta
                        performance_drop = baseline["base_diff_delta"] - ablated_delta
                        sample_drops.append(performance_drop)
                        
                    except Exception as e:
                        print(f"Error in sample {sample_idx} for layer {layer}, head {head}: {e}")
                        continue
                
                # Average across samples
                if sample_drops:
                    avg_drop = np.mean(sample_drops)
                    std_drop = np.std(sample_drops)
                    
                    ablation_results.append({
                        "layer": layer,
                        "head": head,
                        "baseline_diff": baseline["base_diff_delta"],
                        "performance_drop": avg_drop,
                        "std_drop": std_drop,
                        "num_samples": len(sample_drops),
                        "case_type": test_case.case_type
                    })
        
        return ablation_results
    
    def run_complete_analysis(self, num_samples: int = 3, limit: Optional[int] = None):
        """Run complete resample ablation analysis across all test cases."""
        # First collect activations for resampling
        self.collect_activations_for_resampling()
        
        print("="*80)
        print("RESAMPLE ABLATION ANALYSIS FOR SUBJECT-VERB NUMBER AGREEMENT")
        print("="*80)
        
        test_cases = self.create_test_cases(limit=limit)
        print(f"\nRunning resample ablation on {len(test_cases)} test cases...")
        print(f"Model: {self.num_layers} layers × {self.num_heads} heads = {self.num_layers * self.num_heads} total heads")
        print(f"Samples per ablation: {num_samples}\n")
        
        all_ablation_data = []
        self.failed_baselines = []
        valid_cases = 0
        
        with torch.no_grad():
            for test_case in tqdm(test_cases, desc="Running Resample Ablation"):
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
                
                # Run resample ablation
                ablation_results = self.run_resample_ablation_on_case(test_case, baseline, num_samples=num_samples)
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
        print("RESAMPLE ABLATION ANALYSIS REPORT")
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
    
    def save_results(self, filename: str = "resample_ablation_results.json"):
        """Save detailed ablation results to JSON."""
        output_path = __import__("pathlib").Path(__file__).parent / filename
        
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
        output_path = __import__("pathlib").Path(__file__).parent / "resample_ablation_visualization.png"
        
        viz_path = create_ablation_visualizations(
            self.results, 
            self.num_layers, 
            self.num_heads, 
            "Resample Ablation", 
            output_path
        )
        print(f"Visualization saved to {output_path}")
        return viz_path
    
    def create_html_report(self, summary: Dict):
        """Create an interactive HTML report."""
        from core.reporting import create_html_report_generic
        output_path = __import__("pathlib").Path(__file__).parent / "resample_ablation_report.html"
        viz_path = self.create_visualizations()
        
        description = (
            "<strong>Resample Ablation Method:</strong> This analysis applies resample ablation to identify critical attention heads\n"
            "for subject-verb number agreement. Resample ablation replaces a component's output with an activation obtained from a\n"
            "randomly sampled prompt. This ensures the model receives data that is both normal (proper activation distribution) and\n"
            "independent of context. This method tests robustness by averaging results across multiple random samples."
        )
        
        create_html_report_generic(
            results=self.results,
            summary=summary,
            viz_path=viz_path,
            output_path=output_path,
            title="Resample Ablation Analysis Report",
            description_html=description
        )


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="Run Resample Ablation analysis for SVN.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of test cases for a faster run.")
    parser.add_argument("--samples", type=int, default=3, help="Number of resamples per ablation.")
    args = parser.parse_args()

    analyzer = ResampleAblationAnalyzer(seed=42)
    
    analyzer.run_complete_analysis(num_samples=args.samples, limit=args.limit)
    
    # Generate report
    summary = analyzer.generate_report()
    
    # Save results
    analyzer.save_results()
    
    # Create visualizations and HTML report
    print("Generating visualizations and HTML report...")
    analyzer.create_html_report(summary)
    
    print("\n" + "="*80)
    print("✓ Resample Ablation analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
