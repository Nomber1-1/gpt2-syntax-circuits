"""
Shared Ablation Analyzer Base Class
====================================
"""
import torch
from transformer_lens import HookedTransformer
from typing import List, Dict, Optional

class BaseAblationAnalyzer:
    """Base class for all ablation experiments."""
    
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading GPT-2 Small on {self.device}...")
        self.model = HookedTransformer.from_pretrained("gpt2-small", device=self.device)
        self.num_layers = self.model.cfg.n_layers
        self.num_heads = self.model.cfg.n_heads
        self.d_head = self.model.cfg.d_head
        self.results = []
    
    def evaluate_logit_difference(self, logits: torch.Tensor, correct_id: int, incorrect_id: int) -> float:
        """Calculate logit difference: correct_logit - incorrect_logit."""
        final_logits = logits[0, -1, :]
        return (final_logits[correct_id] - final_logits[incorrect_id]).item()
    
    def get_baseline_performance(self, test_case) -> Dict:
        """Get baseline performance without ablation."""
        try:
            correct_id = self.model.to_tokens(test_case.correct_token, prepend_bos=False)[0, 0].item()
            incorrect_id = self.model.to_tokens(test_case.incorrect_token, prepend_bos=False)[0, 0].item()
            
            clean_logits, _ = self.model.run_with_cache(test_case.clean_prompt)
            clean_diff = self.evaluate_logit_difference(clean_logits, correct_id, incorrect_id)
            
            corrupted_logits, _ = self.model.run_with_cache(test_case.corrupted_prompt)
            corrupted_diff = self.evaluate_logit_difference(corrupted_logits, correct_id, incorrect_id)
            
            base_diff_delta = clean_diff - corrupted_diff
            
            return {
                "clean_diff": clean_diff,
                "corrupted_diff": corrupted_diff,
                "base_diff_delta": base_diff_delta,
                "correct_id": correct_id,
                "incorrect_id": incorrect_id,
                "is_valid": base_diff_delta > 0
            }
        except Exception as e:
            print(f"Error in baseline: {e}")
            return None
