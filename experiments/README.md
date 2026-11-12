# DriftLab experiments

The experiment harness lives here. Docs are downstream; the source of truth is
`experiments/results/<exp_id>/`.

## Setup

```bash
uv lock
uv sync
uv run pytest experiments/tests -q
```

## Smoke test (no heavy run)

```bash
uv run python -m experiments.run --exp e02 --seeds 42 --limit 1500 --no-proba
uv run python -m experiments.run --exp e02 --seeds 42 --limit 1500 --no-proba --figure-only
uv run python -m experiments.run --exp e02 --report
```

## Core experiments, in order

```bash
# E01: corrected metric on the Exp-01/02 pilot (DeepRiver; GPU if available)
uv run python -m experiments.run --exp e01

# E02: no-drift control (all models + 8 detectors)
uv run python -m experiments.run --exp e02

# E03: classical ladder (frozen, scheduled, oracle, HT, HAT, ARF, LB, ADWINBagging)
uv run python -m experiments.run --exp e03

# E04: detector × shape × generator matrix (8 detectors, 2 hosts, raw + EWMA)
uv run python -m experiments.run --exp e04

# E05: severity × width sweep
uv run python -m experiments.run --exp e05

# E06: label delay shifts detection by L (Lemma 1)
uv run python -m experiments.run --exp e06

# E07: block-prequential gate (B=1,32,128 vs ARF)
uv run python -m experiments.run --exp e07

# E08: headline DriftFormer vs ARF/HAT/frozen/oracle
uv run python -m experiments.run --exp e08

# E09: pretrain on three generators, evaluate on the held-out one
uv run python -m experiments.pretrain --generators sea,hyperplane,agrawal \
    --out experiments/checkpoints/df_logo_rbf.pt --tasks 8 --samples 20000
uv run python -m experiments.run --exp e09

# E10: ingredient ablations
uv run python -m experiments.run --exp e10

# E11: label swap and feature permutation
uv run python -m experiments.run --exp e11

# E12: architecture sensitivity
uv run python -m experiments.run --exp e12

# E13: calibration under drift (saves probabilities)
uv run python -m experiments.run --exp e13

# E14: real streams (Elec2 downloads once; Phishing is offline)
uv run python -m experiments.run --exp e14

# E15: Delay Tax on the model panel
uv run python -m experiments.run --exp e15

# Derived analyses (no new model runs)
uv run python -m experiments.detector_report   # E04 best detector per shape + host correlation
uv run python -m experiments.schedule_report   # E03 scheduled-retrain phase-noise KS test
uv run python -m experiments.delay_report      # E06 raw delay slope vs L
uv run python -m experiments.cost_report       # E16 samples/s, bytes, quality
```

Every run accepts `--seeds 0-4` to cut the matrix, `--limit N` for a smoke slice,
`--out DIR` to redirect artifacts, and `--figure-only` to re-render from the
committed artifact. `--report` prints the acceptance verdicts and model table.

## Artifact layout

```
experiments/results/<exp_id>/
  meta.json          config, hashes, git SHA, versions
  summary.json       aggregates + acceptance verdicts
  results.json       compatibility summary
  episodes.jsonl     per episode × model (AUE, damage, recovery)
  detections.jsonl   per detector × host × channel × delay (F1, NDT, FAR)
  series.npz         per-sample predictions, errors, rolling kappa, probabilities
  figures/*.png      re-rendered from the artifact only
```

`exp01/` and `exp02/` are the original pilot artifacts and are never modified.
E01 writes the corrected numbers beside them.
