# FRME-GNN — Stage 4

Node classification on **Bail, Credit and German**. This repository contains A0–A5 and runs independently; it extends Stage 3 with prediction knowledge distillation. This code does not download datasets, reconstruct the original baseline splits, or claim results on the three datasets.

## Models

| Run | Training objective |
| --- | --- |
| A0 | Full-edge GCN, cross-entropy |
| A1 | Learned-mask GCN, cross-entropy |
| A2 | A1 + frozen A0 prediction fidelity (KL) + mask sparsity |
| A3 | A2 + soft demographic-parity/equal-opportunity gaps + edge-mask exposure gap |
| A4 | A3 + prediction and explanation-mask consistency under random edge edits |
| **A5** | **A4 + temperature-scaled knowledge distillation from frozen A0** |

A5 adds `lambda_kd * T² * KL(softmax(A0_logits/T) || softmax(A5_logits/T))` on **train nodes only**. `T` is `--kd-temperature` (default 2); `--lambda-kd` defaults to 0.5. The original A2 fidelity term remains in A5; a comparison with `--lambda-kd 0` is therefore essential to establish whether soft-target distillation adds value. A0 is trained separately for each seed and is frozen for A2–A5. Both training and validation follow the same protocol as Stage 3. The mask generator uses local graph context for A1–A5.

The teacher may encode bias: compare group-specific outcomes and do not assume KD improves fairness. No geometric sensitivity-direction distillation is included in this stage.

## Required data

Prepare `data/bail.npz`, `data/credit.npz`, and `data/german.npz` with:

- `x`: node features, float `[N,F]`
- `edge_index`: zero-based edge endpoints, `[2,E]` or `[E,2]`
- `y`: binary target labels `[N]`
- `sensitive`: binary sensitive-group indicator `[N]`
- `train_idx`, `val_idx`, `test_idx`: disjoint node indices

**Match dataset construction, features and original splits** used in NIFTY Vanilla, NIFTY SSF, FairGNN (both variants) and EDITS before comparing with old numbers. Optional per-seed files `bail_seed0.npz` etc. override a fixed split file. The input graph is transductive: training labels are restricted to train nodes, but message passing can traverse test nodes without using their labels.

If the old splits are unavailable, `prepare_splits.py` creates a **fixed, stratified 60/20/20** split from `data/bail_raw.npz` (similarly for Credit and German). Stratification attempts joint label/sensitive-group proportions, falling back to label-only. It preserves natural test distribution, **not artificial class balancing**. In this case, rerun every existing baseline on the new splits; comparing with your original saved scores is invalid.

## Installation and commands

Python 3.10+ recommended. PyTorch CPU works; CUDA can be used when available.

```bash
python -m pip install -r requirements.txt
python prepare_splits.py --dataset bail --data-dir data --split-seed 42  # only if original split unavailable
python train.py --dataset bail --data-dir data --output results/bail
python train.py --dataset credit --data-dir data --output results/credit
python train.py --dataset german --data-dir data --output results/german
python summarize.py results/bail/stage4_results.csv --output results/bail/summary.csv
```

All runs default to seeds `0 1 2 3 4`. For the specific KD ablation, use **different output folders**:

```bash
python train.py --dataset bail --data-dir data --models A0 A4 A5 --output results/bail_kd --lambda-kd 0.5 --kd-temperature 2
python train.py --dataset bail --data-dir data --models A0 A4 A5 --output results/bail_no_kd --lambda-kd 0
```

Training needs original NPZ graphs; it will fail clearly rather than silently generating surrogate dataset data. The script refuses to overwrite a prior `stage4_results.csv`. To train `A5` alone, provide the matching same-seed A0 checkpoints in that output directory.

## Report and comparisons

Run the same seed list, splits, feature processing, edge treatment and validation criterion for A0–A5. Report mean ± sample standard deviation across seeds for:

- **Classification:** F1, ROC-AUC, positive prediction rate.
- **Prediction fairness:** absolute demographic parity and equal opportunity gaps.
- **Mask diagnostics:** mean weight, fraction above 0.5, group incident-edge exposure gap (proxy, not verified explanation fairness).
- **Random-edit stress test:** final / maximum prediction disagreement, KL drift, shared-edge mask L1 drift, perturbed F1.
- **Efficiency:** separately profile runtime if claiming a speed benefit; A5 distillation is not intended as an inference-speed improvement.

Stage 3 robustness testing uses cumulative **random** undirected edge edits; it is **not** a slow-drift optimized attack, is not guaranteed prediction-preserving and offers no formal robustness certificate. No ground-truth motif explanations are present in these three data files. Do not describe mask stability as motif correctness.

The research question for Stage 4 is whether A5 improves A4's F1/AUC without worsening parity, equality, or robustness. Test at minimum `lambda_kd=0` vs positive, and report the full A0–A5 comparison. Keeping the KD teacher fixed avoids silently using validation/test labels to train the student.

## Files

- `train.py`: A0–A5 architecture, training, metrics, checkpoints
- `robustness.py`: cumulative edge-edit test and common-edge mask alignment
- `prepare_splits.py`: optional fixed stratified split generator
- `summarize.py`: grouped mean ± sample SD CSV
- `requirements.txt`: runtime packages

The data and result folders are intentionally empty and gitignored. Do not commit protected/sensitive underlying node records without dataset licensing/privacy checks.
