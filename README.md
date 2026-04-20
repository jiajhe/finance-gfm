# Finance GFM

Phase 1 implementation of the ICML workshop paper prototype:

- Qlib-backed cross-sectional dataset loader
- Factorized Directed Graph (FDG)
- MLP baseline
- IC / weighted-Pearson training objectives
- Single-market training + evaluation pipeline

Run:

```bash
python -m train.train_single --config configs/source_csi300.yaml
```
