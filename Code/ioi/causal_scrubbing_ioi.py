"""
Causal Scrubbing (Circuit Validation) for SVN
=============================================
Validates the circuit discovered by ACDC.
Tests two hypotheses:
1. Faithfulness (Scrub Outside): Ablating everything OUTSIDE the circuit preserves baseline performance.
2. Completeness (Scrub Inside): Ablating only the nodes INSIDE the circuit destroys performance.
"""

import torch
import json
from pathlib import Path
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Set, Tuple
import sys

ROOT = str(Path(__file__).resolve().parent.parent)
if ROOT not in sys.path:
    sys.path.append(ROOT)

from core.base_analyzer import BaseAblationAnalyzer
from core.ioi_dataset import get_ioi_test_cases

class CausalScrubber(BaseAblationAnalyzer):
    """Validates an identified circuit using Causal Scrubbing principles."""
    
    def __init__(self, device=None):
        super().__init__(device=device)

    def load_circuit(self) -> Set[Tuple[int, int]]:
        """Loads the discovered circuit from the ACDC results JSON."""
        results_path = Path(__file__).parent / "acdc_ioi_results" / "acdc_results.json"
        if not results_path.exists():
            raise FileNotFoundError(f"ACDC results not found at {results_path}. Run acdc_ioi.py first.")
        
        with open(results_path, "r") as f:
            data = json.load(f)
            
        ablations = data.get("valid_ablations", [])
        circuit = set()
        for row in ablations:
            if row.get("is_in_circuit"):
                circuit.add((row["layer"], row["head"]))
        return circuit

    def patch_z_conditional(self, z: torch.Tensor, hook, layer_idx: int, 
                            corrupted_cache: Dict, heads_to_patch: Set[Tuple[int, int]]):
        """Hook to patch specific heads with corrupted activations."""
        # Right-align to handle tokenization length differences
        clean_seq_len = z.shape[1]
        corr_seq_len = corrupted_cache[hook.name].shape[1]
        min_len = min(clean_seq_len, corr_seq_len)
        
        for head in range(self.num_heads):
            if (layer_idx, head) in heads_to_patch:
                z[:, -min_len:, head, :] = corrupted_cache[hook.name][:, -min_len:, head, :]
        return z

    def run_scrubbing(self):
        print("="*80)
        print("RUNNING CAUSAL SCRUBBING CIRCUIT VALIDATION")
        print("="*80)
        
        circuit_heads = self.load_circuit()
        all_heads = set((l, h) for l in range(self.num_layers) for h in range(self.num_heads))
        outside_heads = all_heads - circuit_heads
        
        print(f"Loaded circuit with {len(circuit_heads)} heads.")
        print(f"Heads outside circuit: {len(outside_heads)}")
        
        test_cases = get_ioi_test_cases()
        
        results = {
            "baseline_clean": [],
            "scrub_outside": [],
            "scrub_inside": [],
            "baseline_corrupted": []
        }
        
        hook_names = [f"blocks.{l}.attn.hook_z" for l in range(self.num_layers)]
        
        for idx, tc in enumerate(test_cases, 1):
            baseline = self.get_baseline_performance(tc)
            if not baseline or not baseline["is_valid"]:
                continue
                
            correct_id = baseline["correct_id"]
            incorrect_id = baseline["incorrect_id"]
            
            # 1. Baseline Clean
            clean_logits, _ = self.model.run_with_cache(tc.clean_prompt)
            clean_diff = self.evaluate_logit_difference(clean_logits, correct_id, incorrect_id)
            results["baseline_clean"].append(clean_diff)
            
            # 2. Baseline Corrupted (and cache for patching)
            corr_logits, corr_cache = self.model.run_with_cache(tc.corrupted_prompt, remove_batch_dim=False)
            corr_diff = self.evaluate_logit_difference(corr_logits, correct_id, incorrect_id)
            results["baseline_corrupted"].append(corr_diff)
            
            # 3. Scrub Outside (Keep only circuit)
            hooks_outside = [
                (hook_names[l], partial(self.patch_z_conditional, layer_idx=l, 
                                        corrupted_cache=corr_cache, heads_to_patch=outside_heads))
                for l in range(self.num_layers)
            ]
            scrub_outside_logits = self.model.run_with_hooks(tc.clean_prompt, fwd_hooks=hooks_outside)
            scrub_outside_diff = self.evaluate_logit_difference(scrub_outside_logits, correct_id, incorrect_id)
            results["scrub_outside"].append(scrub_outside_diff)
            
            # 4. Scrub Inside (Ablate only circuit)
            hooks_inside = [
                (hook_names[l], partial(self.patch_z_conditional, layer_idx=l, 
                                        corrupted_cache=corr_cache, heads_to_patch=circuit_heads))
                for l in range(self.num_layers)
            ]
            scrub_inside_logits = self.model.run_with_hooks(tc.clean_prompt, fwd_hooks=hooks_inside)
            scrub_inside_diff = self.evaluate_logit_difference(scrub_inside_logits, correct_id, incorrect_id)
            results["scrub_inside"].append(scrub_inside_diff)
            
            print(f"Evaluated case {idx}/{len(test_cases)}")

        # Calculate Averages
        avg_results = {k: sum(v)/len(v) for k, v in results.items() if v}
        
        print("\n" + "="*80)
        print("CAUSAL SCRUBBING RESULTS (Average Logit Difference)")
        print("="*80)
        print(f"1. Baseline Clean (Unpatched):         {avg_results['baseline_clean']:.4f}  <-- Ideal Target")
        print(f"2. Circuit Only (Scrubbed Outside):    {avg_results['scrub_outside']:.4f}  <-- Should remain high (Faithful)")
        print(f"3. Ablated Circuit (Scrubbed Inside):  {avg_results['scrub_inside']:.4f}  <-- Should drop low (Complete)")
        print(f"4. Baseline Corrupted (Fully Patched): {avg_results['baseline_corrupted']:.4f}  <-- Floor Target")
        
        self.plot_results(avg_results, len(circuit_heads))
        
    def plot_results(self, avg_results: Dict[str, float], circuit_size: int):
        sns.set_style("whitegrid")
        plt.figure(figsize=(10, 6))
        
        labels = [
            "Baseline Clean\n(Full Model)", 
            f"Circuit Only\n({circuit_size} heads active)", 
            f"Ablate Circuit\n({circuit_size} heads patched)", 
            "Baseline Corrupted\n(Fully Patched)"
        ]
        values = [
            avg_results['baseline_clean'],
            avg_results['scrub_outside'],
            avg_results['scrub_inside'],
            avg_results['baseline_corrupted']
        ]
        colors = ["#3498db", "#2ecc71", "#e74c3c", "#95a5a6"]
        
        bars = plt.bar(labels, values, color=colors, edgecolor="black", linewidth=1.2)
        
        plt.axhline(0, color="black", linewidth=1)
        plt.ylabel("Exact-Match Logit Difference", fontsize=12, fontweight="bold")
        plt.title("Causal Scrubbing: Circuit Validation", fontsize=16, fontweight="bold", pad=20)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.1),
                     f'{height:.3f}', ha='center', va='bottom' if height >= 0 else 'top', 
                     fontweight='bold', fontsize=11)
            
        output_path = Path(__file__).parent / "causal_scrubbing_results.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"\nVisualization saved to {output_path}")

if __name__ == "__main__":
    scrubber = CausalScrubber()
    scrubber.run_scrubbing()