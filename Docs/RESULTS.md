# Project Results and Log

## Latest Comprehensive Dataset Evaluation (Subject-Verb Agreement)
**Model**: `gpt2-small`
**Dataset**: A comprehensive, balanced dataset with 162 unique test cases for Subject-Verb Number Agreement, generated across multiple linguistic permutations.
**Interventions**: Zero, Mean, and Resample Ablation (23,328 records each) and ACDC Iterative Pruning (144 records).

### Side-by-Side Comparison Summary
| Method | Mean Drop | Median Drop | Std Dev | Min Drop | Max Drop | Top Head | Top Head Drop | Pos. Count | Neg. Count |
|--------|-----------|-------------|---------|----------|----------|----------|---------------|------------|------------|
| **Zero Ablation** | 0.0105 | 0.0026 | 0.2746 | -5.0625 | 11.3492 | **L0H9** | 11.3492 | 12114 | 11214 |
| **Mean Ablation** | 0.0150 | 0.0008 | 0.1921 | -1.2180 | 4.0414 | **L11H8** | 4.0414 | 12363 | 10965 |
| **Resample Ablation** | 0.0205 | 0.0012 | 0.2623 | -5.8568 | 7.7260 | **L1H10** | 7.7260 | 12490 | 10838 |
| **ACDC Pruning** | 0.2829 | 0.2927 | 0.0461 | 0.1948 | 0.5318 | **L1H10** | 0.5318 | 144 | 0 |

### Detailed Insights per Method

1.  **Zero Ablation (Deletion)**: This method completely silences a head's output. It yields the lowest mean performance drop, suggesting the model has redundant pathways to compensate for a simple loss of signal. Its top-performing head, `L0H9`, is likely a foundational component whose removal disrupts the basis for later computations.

2.  **Mean Ablation (Confusion via Neutrality)**: This method replaces a head's output with its dataset-average activation. By passing "neutral" information, it actively confuses downstream components. It identifies a late-layer head, `L11H8`, as most critical, suggesting this head is a "Logit Mover" that directly influences the final output probability.

3.  **Resample Ablation (Confusion via Misinformation)**: This method replaces activations with those from a corrupted prompt (e.g., patching a "plural" context with a "singular" one). This "lies" to the model and causes the highest mean performance drop among the standard ablations, pointing to the importance of early-layer head `L1H10` in establishing the correct grammatical context.

4.  **ACDC Iterative Pruning**: Unlike single-head ablations, ACDC evaluates heads in the context of an already-pruned graph. The high mean drop (0.2829) reflects the average importance of each head at the moment it was tested. It identifies `L1H10` as the most critical head, suggesting that establishing correct grammatical context early is the most non-redundant function in the minimal circuit.

---

## Verifying Circuit Validity with Causal Scrubbing (Subject-Verb Agreement)
To validate the circuit discovered by ACDC, we performed causal scrubbing. This method tests two key properties:
*   **Faithfulness**: By ablating all heads *outside* the circuit, we test if the circuit alone is sufficient to perform the task. The model retained **90.67%** of its baseline performance, demonstrating high faithfulness.
*   **Completeness**: By ablating only the heads *inside* the circuit, we test if the circuit is necessary for the task. Performance dropped by **44.40%**, demonstrating moderate completeness and suggesting the presence of redundant pathways (e.g., in MLP layers) that can partially compensate for the circuit's removal.

These results provide strong evidence that the ACDC-discovered circuit is a meaningful and accurate representation of the mechanism for subject-verb agreement in gpt2-small.