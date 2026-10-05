# Roadmap and status

## Done

- [x] Clean / corrupted prompt-pair dataset for subject–verb number agreement
- [x] Shared ablation harness with pluggable intervention types
- [x] Common HTML + JSON reporting layer
- [x] Ablation methods — zero, mean and resample
- [x] ACDC automatic circuit discovery
- [x] Causal scrubbing validation of the discovered circuit
- [x] Confirmed sufficiency (ablating everything outside the circuit preserves baseline
      behaviour) and measured necessity (intervening inside degrades it)
- [x] Second behaviour implemented: indirect object identification, with its own dataset,
      ACDC run and causal scrubbing

## Next

- [ ] Extend circuit discovery past attention heads into MLP blocks. The completeness result
      (44.40%) points there — something outside the circuit compensates.
- [ ] Per-layer degradation plots, modelling how performance falls off across layers and heads.
- [ ] Paper-format write-up.
