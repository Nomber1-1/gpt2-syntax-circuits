# Mechanistic Interpretability of Subject–Verb Agreement in GPT-2 Small

Locating **and then validating** the internal circuit that `gpt2-small` uses for subject–verb
number agreement, by combining four ablation methods, automatic circuit discovery (ACDC), and
causal scrubbing.

**The headline result:** a minimal circuit that retains **90.67%** of baseline accuracy when
every attention head outside it is ablated, and loses **44.40%** when the circuit itself is
removed.

Built as a three-person team project for COMP 579 at McGill University (Winter 2026). The topic
was our own proposal, not an assigned task.

---

## Why this is more than an ablation study

Most "which head matters?" studies stop at a ranking. That only shows a component is
*involved* — not that it is *sufficient* on its own, nor that the rest of the network cannot
route around it. This project treats those as two separate, measurable properties.

It also shows why the choice of intervention matters. The three single-head methods disagree
about which head is most important, because each perturbs the network in a different way:

| Method | What it does | Mean drop | Top head |
|---|---|---|---|
| **Zero ablation** | Deletes a head's output entirely | 0.0105 | `L0H9` (11.3492) |
| **Mean ablation** | Replaces output with the dataset-average activation | 0.0150 | `L11H8` (4.0414) |
| **Resample ablation** | Replaces output with activations from a corrupted prompt | 0.0205 | `L1H10` (7.7260) |
| **ACDC pruning** | Scores each head *inside an already-pruned graph* | 0.2829 | `L1H10` (0.5318) |

Each method carries its own reading:

- **Zero ablation** produces the smallest mean drop, which suggests the model can compensate
  for a simply-silenced signal through redundant pathways.
- **Mean ablation** passes "neutral" information downstream, which actively confuses later
  components, and points at a late-layer head that appears to move the output logits directly.
- **Resample ablation** "lies" to the model by substituting a plural context into a singular
  prompt, and produces the largest drop of the three — implicating an **early** head in
  establishing grammatical context at all.
- **ACDC** answers a different question. Rather than scoring heads in isolation, it measures
  each head's importance at the moment it is tested during iterative pruning, so it captures
  non-redundant function.

**Two independent methods that measure non-redundant function — resample ablation and ACDC —
converge on `L1H10`.** That convergence, not any single ranking, is the actual finding: for this
task the least redundant thing the model does is establish grammatical context early.

## Validation: causal scrubbing

A circuit is only a hypothesis until you test it. We tested both properties:

- **Faithfulness — 90.67% of baseline retained.** Ablating every head *outside* the circuit
  barely hurts, so the circuit is *sufficient* to perform the task.
- **Completeness — 44.40% drop.** Ablating the heads *inside* the circuit degrades but does not
  destroy performance, so the circuit is only moderately *necessary*.

The low completeness is reported as a genuine limitation rather than smoothed over: it implies
redundant pathways elsewhere — most likely in MLP layers — can partially compensate when the
circuit is removed.

## Method details

- **Model:** `gpt2-small`, via `transformer_lens`.
- **Task:** subject–verb number agreement over a balanced dataset of **162 unique test cases**
  generated across multiple linguistic permutations.
- **Scale:** 23,328 records for each of the zero, mean and resample ablations; 144 for ACDC
  iterative pruning.
- **Metrics:** exact-match accuracy, with mean / median / std-dev / min / max per-head drop.
- **Secondary task:** the same pipeline is also wired for **indirect object identification
  (IOI)**, with its own dataset, ACDC run and causal scrubbing (`Code/ioi/`). The reported
  results above are the subject–verb agreement track.

## Repository layout

```
Code/
├── core/                    Shared infrastructure
│   ├── base_analyzer.py     Ablation harness
│   ├── reporting.py         HTML + JSON report generation
│   ├── svn_dataset.py       Subject–verb agreement prompt pairs
│   └── ioi_dataset.py       IOI prompt pairs
├── svn/                     Subject–verb agreement track
│   ├── master_test.py       Runs the full ablation suite
│   ├── zero_ablation/       Script + results + visualisation
│   ├── mean_ablation/
│   ├── resample_ablation/
│   ├── acdc_svn.py          Automatic circuit discovery
│   ├── causal_scrubbing_svn.py
│   ├── compare_ablations.py Cross-method comparison table + figure
│   └── svn_testing/         Baseline accuracy on the clean dataset
└── ioi/                     Indirect object identification track
    ├── acdc_ioi.py
    ├── causal_scrubbing_ioi.py
    └── ioi_testing/

Docs/
├── RESULTS.md               Full quantitative results and interpretation
└── NEXT_STEPS.md            Task list and what we would do next
```

## Running it

```bash
conda create -n interp python=3.12
conda activate interp

pip install transformer_lens
pip install -r requirements.txt

# Run the full ablation suite (zero, mean, resample, ACDC) for subject–verb agreement
python Code/svn/master_test.py
```

`graphviz` is required for circuit visualisation and may need a system-level install
(`apt-get install graphviz`, `brew install graphviz`, or the Windows installer).

## Outputs

Every run produces reports you can read without re-running anything:

- `*_report.html` — per-head results with charts, openable in a browser
- `*_results.json` — the same numbers, machine-readable
- `*_visualization.png` — head-importance plots
- `acdc_circuit_dag.png` — the discovered circuit as a directed graph
- `compare_results/ablation_comparison_table.tex` — the cross-method table, ready for LaTeX

## Team

This was a three-person team project; the work was shared across the group.

## Licence

MIT — see [LICENSE](LICENSE).
