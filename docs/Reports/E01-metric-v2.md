# E01 — Metric v2 correction (Exp-01/02 revision)

**Status:** concluded (clean three-model panel: DeepRiver, frozen RF, timing
oracle) · **Artifact:** `experiments/results/e01/` ·
**Command:** `uv run python -m experiments.run --exp e01`

---

## 1. Why this experiment exists

Exp-01 and Exp-02 reported a **294×** mean damage gap and a **24×** worst-vs-best
gap. Both were inflated by a defect in the metric, not by the models:

- `damage_area` integrated from `τ` to the *recovery point*.
- On 8 of 11 seeds the recovery rule collapsed to `recovery = τ` (the model never
  left its band), so the integral covered a **single sample**.
- Streaming damage was therefore **truncated**; frozen damage — which never
  recovered — was integrated over the full window.
- Both numbers were **upper bounds**, and the viva brief already flagged the
  defect as a known weakness.

E01 re-runs the same pilot with a metric that cannot be gamed this way.

## 2. What metric v2 changes

| Metric | Definition | Why |
|---|---|---|
| **`damage_ref`** (primary) | `Σ max(κ_ref − κ(t), 0)` over a fixed window from `τ` | `κ_ref` is shared across the model panel, so a weak model cannot score zero damage by having a low own-baseline |
| **`damage_full`** | Same, from `τ` to the end of the stream | Prices the whole remaining stream, not just the pre-registered window |
| **`AUE`** | mean error minus the timing-oracle error over the fixed window | Dimensionless; oracle is an RF refit at `τ` on pre-drift data |
| **`T_exit` / `T_recover`** | Recovery only counts after a real band exit | Removes the “0 = instant recovery” artefact |
| **`recovery_rate`** | Fraction of seeds with a real recovery | Replaces the misleading mean of a floored recovery time |

The fixed window is **5,000 samples** for the abrupt SEA stream. The recovery
band is `0.95·κ_ref` with a 500-sample stretch.

## 3. Setup

| | |
|---|---|
| Stream | SEA, one continuous `U(0,10)³` draw |
| Length | 60,000 samples · warm-up 12,000 · drift at `τ = 30,000` |
| Drift | abrupt: `x₀ + x₁ > 8` → `x₀ + x₁ > 9` |
| Noise | 10% label noise |
| Seeds | 11: `42, 0…9` |
| Streaming | DeepRiver `RollingClassifier`, RNN 3→16→2, window 500 |
| Frozen | `RandomForestClassifier(100)`, fit once on the warm-up |
| Oracle | RF refit at `τ` on the last 12k pre-drift samples (knows *when*) |

The panel is DeepRiver, frozen RF, and the timing oracle. The earlier artifact
that also contained HT/HAT — caused by a host-default bug that has since been
fixed — is archived at `experiments/results/_archive/e01_5model/`; every number
below comes from the clean rerun.

## 4. Results

### 4.1 Aggregate, 11 seeds

| Metric | DeepRiver | Frozen RF | Timing oracle |
|---|---:|---:|---:|
| Full-run accuracy | **0.8830 ± 0.0021** | 0.8539 ± 0.0017 | 0.8543 ± 0.0016 |
| Full-run Cohen κ | **0.7530 ± 0.0044** | 0.6850 ± 0.0035 | 0.6860 ± 0.0033 |
| AUE | **−0.0472 ± 0.0058** | +0.0010 ± 0.0027 | 0 (reference) |
| **`damage_ref`** (5k window) | **74.8 ± 61.8** | **515.0 ± 97.1** | 505.0 ± 97.6 |
| `damage_full` (rest of stream) | **321.4 ± 203.8** | **3,472.9 ± 559.7** | 3,425.4 ± 506.8 |
| Recovery rate | **8/11 (0.727)** | **0/11 (0.000)** | 0/11 |
| Recovery time, recovered seeds | 285–4,279; mean 1,502 | never | never |

![E01 metric v2 — damage_ref and AUE per model](../Phase1:Experimentation/figures/e01-metric-v2-model-comparison.png)

*The two panels are deliberately separate: `damage_ref` is the primary damage
metric, AUE is the excess over the timing oracle. AUE is negative for DeepRiver
because the timing oracle is weak (its `damage_ref` is 505, almost identical to
frozen), not because DeepRiver is impossible.*

### 4.2 Per-seed `damage_ref`

| Seed | DeepRiver | Frozen RF |
|---:|---:|---:|
| 42 | 51.7 | 514.2 |
| 0 | 25.1 | 402.6 |
| 1 | 13.1 | 498.5 |
| 2 | 46.1 | 509.5 |
| 3 | 159.6 | 565.9 |
| 4 | 55.4 | 522.5 |
| 5 | 80.1 | 578.2 |
| 6 | 24.8 | 295.2 |
| 7 | 71.3 | 586.7 |
| 8 | 77.9 | 531.9 |
| 9 | 217.9 | 660.0 |
| **mean ± SD** | **74.8 ± 61.8** | **515.0 ± 97.1** |

DeepRiver's damage is lower on **11/11** seeds. With the clean panel the
worst DeepRiver seed (217.9) sits below the best frozen seed (295.2) — **0.74×**
its damage. The mean gap is **6.9×**, and the full-stream gap is **10.8×**.

## 5. Verdict

| Pre-registered condition | Required | Measured | Verdict |
|---|---|---|---|
| Frozen never recovers | every seed | 0/11 | **PASS** |
| DeepRiver `damage_ref` lower | ≥10/11 seeds | 11/11 | **PASS** |
| AUE vs timing oracle | ratio or better | DeepRiver −0.0472, frozen +0.0010 | **PASS** — mechanism, not upper bound |

**The claim survives; the magnitude shrinks.** A streaming model that keeps
learning pays about **6.9× less damage** in the pre-registered window than a
frozen model (**10.8×** over the rest of the stream), and the frozen model never
recovers. The old **294×** and **24×** are retired.

## 6. What this retires

- **294× mean damage gap** → corrected to **6.9×** (`damage_ref`) or **10.8×**
  (`damage_full`) on the clean panel.
- **24× worst-vs-best** → retired. The worst DeepRiver seed is **0.74×** the best
  frozen seed; the worst-case claim was a truncation artefact. The earlier
  four-model panel's 1.27× is superseded by the clean rerun.
- **“835 samples”** → seed 42 still recovers in 835 samples under the corrected
  rule, but recovery is **8/11**, with values **285–4,279**. The number was seed
  luck; the distribution is the result.
- **“Damage area and dip depth are the trustworthy metrics”** → too generous.
  `damage_ref` is the trustworthy one; `damage_full` is the one that prices the
  whole stream; recovery time is meaningful only conditional on an exit.

## 7. Caveats to keep in the report

1. **Panel composition (resolved).** The panel is DeepRiver, frozen RF, and the
   timing oracle. The pre-fix four-model artifact is kept at
   `experiments/results/_archive/e01_5model/` for auditability; the numbers here
   are the clean rerun and supersede it.
2. **AUE is not an upper bound.** The timing oracle knows *when* the drift
   happened but is refit only on pre-drift data; its damage is almost identical
   to frozen. AUE therefore measures “excess over a weak oracle,” not “distance
   from the best possible model.” Use `damage_ref` as primary.
3. **One synthetic stream.** This is still SEA, one abrupt drift. The detector
   matrix, real streams, and the Delay Tax are the next gates.

## 8. Reproduce

```bash
uv run python -m experiments.run --exp e01
uv run python -m experiments.run --exp e01 --report
uv run python -m experiments.run --exp e01 --figure-only
```

**Artifacts:** `experiments/results/e01/summary.json`,
`experiments/results/e01/models.jsonl`, `experiments/results/e01/episodes.jsonl`,
`experiments/results/e01/series.npz`.
