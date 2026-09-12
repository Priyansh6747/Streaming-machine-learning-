"""Exp-02: seed sweep — is Exp-01's 835-sample recovery a fluke of one seed?

Spec:      docs/Reports/Results.md (Exp-02 row) · docs/Reports/Sept9.md (next milestone 1)
Protocol:  docs/Phase1:Experimentation/streaming-vs-batch-metrics.md §5 (seeds & statistics)
Baseline:  docs/Phase1:Experimentation/exp01-deepriver-vs-frozen-sea.md (seed 42, single run)

Exp-01 was n = 1. Synthetic streams are random (feature draws + 10% label noise), so a
single run cannot separate signal from noise (thesis F8, Demsar 2006). This experiment
reruns the *same* stream, the *same* two models and the *same* metric definitions as
Exp-01 and changes exactly one thing: the RNG seed.

Acceptance bar (pre-registered in Results.md §4):
  * recovery time is reported as a distribution (mean +/- SD) over >= 10 seeds, not a
    single number;
  * the frozen model's "never recovers" holds for every seed (it is also the check that
    there was real drift in every run).

Run (CUDA):
  uv run experiments/scripts/exp02_seed_sweep_sea.py --require-cuda
"""
from __future__ import annotations

import argparse
import json
import resource
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from river.metrics import Accuracy, CohenKappa
from river.utils import Rolling
from sklearn.ensemble import RandomForestClassifier

# --- Scenario constants — identical to Exp-01 (do not drift these) -------------
N_TOTAL = 60_000        # stream length (samples)
WARM_UP = 12_000        # frozen model's training window (20%)
TAU = 30_000            # drift onset (samples)
NOISE = 0.1             # 10% label noise
WINDOW = 500            # DeepRiver rolling window (pre-registered default)
LR = 1e-3               # DeepRiver learning rate (pre-registered default)
ROLL_W = 1_000          # rolling-Kappa window
RECOVERY_BAND = 0.95    # recovery = rolling Kappa >= band * kappa0
RECOVERY_LOOKAHEAD = 5_000  # samples after tau in which recovery must happen
RECOVERY_STRETCH = 500       # consecutive samples at/above the band to count

# 11 seeds (>= the 10 the acceptance bar asks for); 42 is Exp-01's seed, rerun here as
# a reproducibility cross-check against the committed Exp-01 artifact.
DEFAULT_SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


def _fmt(v, spec: str = ".3f") -> str:
    """Format a possibly-None number for the progress log."""
    return "n/a" if v is None else format(v, spec)


class _RNNHead(torch.nn.Module):
    """Small RNN over the rolling window, Linear head to the two classes.

    Identical to Exp-01's module. Input: (window_size, batch=1, n_features).
    DeepRiver's RollingClassifier requires the first trainable layer to take
    n_features units (its feature-incremental padding logic depends on it), hence
    an RNN head rather than a flattened MLP.
    """

    def __init__(self, n_features: int, n_hidden: int = 16) -> None:
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=n_features, hidden_size=n_hidden, batch_first=True
        )
        self.head = torch.nn.Linear(n_hidden, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(X.permute(1, 0, 2))  # (seq, 1, feat) -> (1, seq, hidden)
        return self.head(out[:, -1, :])         # last timestep -> logits


def build_stream(seed: int, n_total: int = N_TOTAL,
                 tau: int = TAU) -> list[tuple[dict[int, float], int]]:
    """One continuous SEA feature stream; the label rule switches abruptly at tau.

    Same construction as Exp-01 §3a: river 0.25's SEA reseeds its internal RNG per
    generator, so concatenating two variants replays identical feature vectors under
    the new label rule (a train-set memorisation confound). Instead: one uninterrupted
    draw of x ~ U(0,10)^3, labelled by concept 0 (sum > 8) before tau and concept 1
    (sum > 9) from tau on. Only `seed` differs between runs.
    """
    rng = np.random.default_rng(seed)
    n_pre = min(tau, n_total)
    out: list[tuple[dict[int, float], int]] = []
    for t in range(n_total):
        x = {i: float(rng.uniform(0.0, 10.0)) for i in range(3)}
        label = x[0] + x[1] > (8.0 if t < n_pre else 9.0)
        if rng.random() < NOISE:
            label = not label
        out.append((x, int(label)))
    return out


def make_deepriver(seed: int, device: str = "cpu") -> "RollingClassifier":
    from deep_river.classification import RollingClassifier

    class _TrainableRollingClassifier(RollingClassifier):
        """deep-river 0.3.4 runs the forward before `module.train()` in `_learn`,
        so after a predict (eval mode) cuDNN RNN backward errors out. Force train
        mode before the update step. (Same workaround as Exp-01 §3b.)"""

        def learn_one(self, x, y):  # noqa: D102
            self.module.train()
            return super().learn_one(x, y)

    return _TrainableRollingClassifier(
        module=_RNNHead(n_features=3, n_hidden=16),
        loss_fn="cross_entropy",
        optimizer_fn="adam",
        lr=LR,
        device=device,
        seed=seed,
        window_size=WINDOW,
        is_class_incremental=False,
        is_feature_incremental=False,
    )


def drift_metrics(series_t: list[int], k_dr: list[float | None], k_fr: list[float | None],
                  tau: int = TAU) -> dict:
    """kappa0, dip depth, recovery time and damage area per model (Exp-01 doc §4).

    Definitions are copied verbatim from the Exp-01 script so Exp-02 is a sweep of
    one experiment, not a new experiment.
    """

    def band_and_recovery(k: list[float | None]) -> dict:
        t = np.asarray(series_t, dtype=np.int64)
        kappa = np.array([np.nan if v is None else v for v in k], dtype=float)
        pre = (t >= tau - 1_000) & (t < tau)
        kappa0 = float(np.nanmean(kappa[pre])) if np.any(pre) else None
        post = (t >= tau) & (t < tau + RECOVERY_LOOKAHEAD)
        dip = float(np.nanmin(kappa[post])) if np.any(post) else None
        depth = (kappa0 - dip) if kappa0 is not None and dip is not None else None
        # recovery: first sample from which rolling Kappa stays >= band for 500
        # consecutive samples within [tau, tau + 5000)
        recover: int | None = None
        if kappa0 is not None:
            band = RECOVERY_BAND * kappa0
            run = 0
            for i in np.where(post)[0]:
                if np.isfinite(kappa[i]) and kappa[i] >= band:
                    run += 1
                    if run >= RECOVERY_STRETCH:
                        recover = int(t[i] - RECOVERY_STRETCH + 1)
                        break
                else:
                    run = 0
        # damage area: sum(max(kappa0 - kappa, 0)) from tau to recovery (stream end if never)
        area: float | None = None
        if kappa0 is not None:
            idx = np.where(t >= tau)[0]
            if recover is not None:
                idx = idx[t[idx] <= recover]
            area = float(np.sum(np.maximum(kappa0 - kappa[idx], 0.0)))
        return {
            "kappa0": kappa0,
            "min_kappa": dip,
            "dip_depth": depth,
            "recovery_time": recover,
            "recovery_samples": (recover - tau) if recover is not None else None,
            "damage_area": area,
        }

    return {
        "window": "t in [tau, tau+5000) for dip; recovery within same window",
        "deepriver": band_and_recovery(k_dr),
        "frozen": band_and_recovery(k_fr),
    }


def run_one_seed(seed: int, n_total: int = N_TOTAL, warm_up: int = WARM_UP,
                 tau: int = TAU, device: str = "cpu") -> dict:
    """One Exp-01 replica under `seed`: same stream shape, same models, same protocol."""
    # Determinism: seed Python/numpy (stream + RF) and torch (net init + shuffling)
    # before anything stochastic is built for this replica.
    torch.manual_seed(seed)
    np.random.seed(seed)

    stream = build_stream(seed, n_total, tau)
    n = len(stream)

    # --- frozen batch model: fit once on the warm-up window --------------------
    X_all = np.array([list(x.values()) for x, _ in stream], dtype=np.float64)
    y_all = np.array([y for _, y in stream], dtype=np.int64)
    t_fit = time.perf_counter()
    frozen = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    frozen.fit(X_all[:warm_up], y_all[:warm_up])
    t_fit = time.perf_counter() - t_fit
    # batch-predict the whole stream once (frozen model never learns after warm-up)
    pred_frozen = frozen.predict(X_all).astype(int)

    # --- streaming model -------------------------------------------------------
    dr = make_deepriver(seed, device)

    # --- prequential loop over the post-warm-up region -------------------------
    acc_dr, acc_fr = Accuracy(), Accuracy()
    kappa_dr, kappa_fr = CohenKappa(), CohenKappa()
    roll_dr = Rolling(CohenKappa(), window_size=ROLL_W)
    roll_fr = Rolling(CohenKappa(), window_size=ROLL_W)

    t_start = time.perf_counter()
    series_t: list[int] = []
    k_dr: list[float | None] = []
    k_fr: list[float | None] = []
    for t, (x, y) in enumerate(stream):
        if t < warm_up:
            # warm-up: only DeepRiver learns (rolling window must be warm by tau)
            dr.learn_one(x, y)
            continue
        # test
        yp_dr = dr.predict_one(x)
        acc_dr.update(y, yp_dr)
        kappa_dr.update(y, yp_dr)
        roll_dr.update(y, yp_dr)
        yp_fr = int(pred_frozen[t])
        acc_fr.update(y, yp_fr)
        kappa_fr.update(y, yp_fr)
        roll_fr.update(y, yp_fr)
        series_t.append(t)
        k_dr.append(roll_dr.get())
        k_fr.append(roll_fr.get())
        # train
        dr.learn_one(x, y)
        if (t - warm_up) % 20_000 == 0 and (t - warm_up) > 0:
            print(f"      seed {seed}: {t - warm_up:,}/{n - warm_up:,} evaluated "
                  f"({(t - warm_up) / (time.perf_counter() - t_start):.0f} samples/s)",
                  flush=True)
    t_run = time.perf_counter() - t_start

    n_eval = n - warm_up
    return {
        "seed": seed,
        "drift_metrics": drift_metrics(series_t, k_dr, k_fr, tau),
        "full_run": {
            "n_evaluated": n_eval,
            "deepriver": {"accuracy": acc_dr.get(), "cohen_kappa": kappa_dr.get()},
            "frozen": {"accuracy": acc_fr.get(), "cohen_kappa": kappa_fr.get()},
        },
        "cost": {
            "deepriver_loop_s": t_run,
            "frozen_fit_s": t_fit,
            "samples_per_s": (n_eval / t_run) if t_run else None,
        },
    }


def _summary(values: list[float | None]) -> dict:
    """mean +/- SD (and spread) over the non-None entries; counts Nones separately."""
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "median": None}
    return {
        "n": len(vals),
        "mean": statistics.mean(vals),
        "sd": statistics.stdev(vals) if len(vals) >= 2 else None,
        "min": min(vals),
        "max": max(vals),
        "median": statistics.median(vals),
    }


def aggregate(per_seed: list[dict]) -> dict:
    """Collapse the per-seed replicas into the distribution the acceptance bar asks for."""
    dr = [r["drift_metrics"]["deepriver"] for r in per_seed]
    fr = [r["drift_metrics"]["frozen"] for r in per_seed]

    dr_rec = [m["recovery_samples"] for m in dr]
    fr_rec = [m["recovery_samples"] for m in fr]
    dr_area = [m["damage_area"] for m in dr]
    fr_area = [m["damage_area"] for m in fr]

    def with_never(summary: dict, values: list[float | None]) -> dict:
        return {**summary, "never": sum(1 for v in values if v is None)}

    return {
        "n_seeds": len(per_seed),
        "streaming_recovery_samples": with_never(_summary(dr_rec), dr_rec),
        "frozen_recovery_samples": with_never(_summary(fr_rec), fr_rec),
        "streaming_dip_depth": _summary([m["dip_depth"] for m in dr]),
        "frozen_dip_depth": _summary([m["dip_depth"] for m in fr]),
        "streaming_kappa0": _summary([m["kappa0"] for m in dr]),
        "frozen_kappa0": _summary([m["kappa0"] for m in fr]),
        "streaming_min_kappa": _summary([m["min_kappa"] for m in dr]),
        "frozen_min_kappa": _summary([m["min_kappa"] for m in fr]),
        "streaming_damage_area": _summary(dr_area),
        "frozen_damage_area": _summary(fr_area),
        "full_run": {
            "deepriver_accuracy": _summary([r["full_run"]["deepriver"]["accuracy"] for r in per_seed]),
            "deepriver_cohen_kappa": _summary([r["full_run"]["deepriver"]["cohen_kappa"] for r in per_seed]),
            "frozen_accuracy": _summary([r["full_run"]["frozen"]["accuracy"] for r in per_seed]),
            "frozen_cohen_kappa": _summary([r["full_run"]["frozen"]["cohen_kappa"] for r in per_seed]),
        },
        "total_deepriver_runtime_s": sum(r["cost"]["deepriver_loop_s"] for r in per_seed),
    }


def acceptance(agg: dict) -> dict:
    """The pre-registered bar from Results.md §4, evaluated against this sweep."""
    n = agg["n_seeds"]
    return {
        "min_seeds_10": {"required": 10, "actual": n, "pass": n >= 10},
        "streaming_recovers_on_every_seed": {
            "recovered": agg["streaming_recovery_samples"]["n"], "of": n,
            "pass": agg["streaming_recovery_samples"]["n"] == n,
        },
        "frozen_never_recovers_on_every_seed": {
            "never": agg["frozen_recovery_samples"]["never"], "of": n,
            "pass": agg["frozen_recovery_samples"]["never"] == n,
        },
    }


def read_exp01_recovery(repo: Path) -> int | None:
    """Exp-01's committed seed-42 recovery time, for the reproducibility cross-check."""
    path = repo / "experiments" / "exp01" / "results.json"
    if not path.exists():
        return None
    try:
        return (json.loads(path.read_text())
                .get("drift_metrics", {}).get("deepriver", {}).get("recovery_samples"))
    except (json.JSONDecodeError, AttributeError):
        return None


def cross_check(per_seed: list[dict], exp01_recovery: int | None) -> dict:
    """Does the seed-42 replica land on the committed Exp-01 number?"""
    seed42 = next((r for r in per_seed if r["seed"] == 42), None)
    got = seed42["drift_metrics"]["deepriver"]["recovery_samples"] if seed42 else None
    return {
        "seed": 42,
        "exp01_recovery_samples": exp01_recovery,
        "exp02_recovery_samples": got,
        "delta_samples": (got - exp01_recovery)
                        if got is not None and exp01_recovery is not None else None,
        "matches": (got == exp01_recovery) if got is not None and exp01_recovery is not None else None,
        "note": ("Exact match not required (cuDNN RNN backward is not bit-deterministic); "
                 "the check is that the same seed lands in the same regime."),
    }


def make_figures(per_seed: list[dict], figures_dir: Path) -> list[Path]:
    """Render the Exp-02 figure suite straight from the artifact.

    Every figure stands on its own (one story per figure, never two stories crammed
    into one axes pair):

      1. exp02-seed-sweep-recovery.png   overview: recovery + damage per seed
      2. exp02-recovery-per-seed.png     the recovery distribution and its floor
      3. exp02-damage-per-seed.png       damage paid per seed, both models
      4. exp02-damage-distribution.png   the damage gap as a distribution
      5. exp02-dip-depth-per-seed.png    how hard the drift hit each model
      6. exp02-kappa-collapse.png        each model's fall from its own baseline
      7. exp02-full-run-kappa.png        full-run Cohen Kappa per seed
      8. exp02-full-run-accuracy.png     full-run accuracy per seed
      9. exp02-recovery-vs-dip.png       why the recovery metric under-reports
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    figures_dir.mkdir(parents=True, exist_ok=True)
    labels = [str(r["seed"]) for r in per_seed]
    x = np.arange(len(per_seed))
    n = len(per_seed)
    dr = [r["drift_metrics"]["deepriver"] for r in per_seed]
    fr = [r["drift_metrics"]["frozen"] for r in per_seed]
    dr_rec = [m["recovery_samples"] for m in dr]
    dr_area = [m["damage_area"] for m in dr]
    fr_area = [m["damage_area"] for m in fr]
    dr_dip = [m["dip_depth"] for m in dr]
    fr_dip = [m["dip_depth"] for m in fr]
    dr_k0 = [m["kappa0"] for m in dr]
    fr_k0 = [m["kappa0"] for m in fr]
    dr_mink = [m["min_kappa"] for m in dr]
    fr_mink = [m["min_kappa"] for m in fr]
    dr_kap = [r["full_run"]["deepriver"]["cohen_kappa"] for r in per_seed]
    fr_kap = [r["full_run"]["frozen"]["cohen_kappa"] for r in per_seed]
    dr_acc = [r["full_run"]["deepriver"]["accuracy"] for r in per_seed]
    fr_acc = [r["full_run"]["frozen"]["accuracy"] for r in per_seed]
    floor = 1e-3
    written: list[Path] = []

    def sd(values) -> float:
        """Sample SD, or NaN when there are too few points (single-seed dev slices)."""
        return statistics.stdev(values) if len(values) >= 2 else float("nan")

    def ratio(num: float, den: float, fmt: str = ".1f") -> str:
        """Formatted ratio that degrades gracefully when the denominator is 0."""
        return format(num / den, fmt) if den else "n/a"

    def seed_axis(ax) -> None:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_xlabel("RNG seed")
        ax.grid(alpha=0.25, lw=0.5)

    def save(fig, name: str) -> None:
        fig.tight_layout()
        path = figures_dir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"saved figure: {path}")

    def zero_stubs(ax, xs, width: float) -> None:
        """Zero damage cannot be drawn on a log axis — stand a hatched stub on the floor."""
        if xs:
            ax.bar(xs, [floor * 1.6] * len(xs), width=width, bottom=floor,
                   color="tab:blue", hatch="///", edgecolor="white", lw=0, zorder=3)

    def new_ax(w: float = 9.5, h: float = 5.2):
        fig, ax = plt.subplots(figsize=(w, h), dpi=150)
        return fig, ax

    # --- 1. overview: recovery (left) + damage (right) --------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4), dpi=150)
    plotted = np.array([v if v is not None else np.nan for v in dr_rec], dtype=float)
    ax1.scatter(x, plotted, s=55, color="tab:blue", zorder=4, label="streaming recovery")
    recovered = [v for v in dr_rec if v is not None]
    mean = statistics.mean(recovered) if recovered else float("nan")
    median = statistics.median(recovered) if recovered else float("nan")
    ax1.axhline(mean, color="tab:blue", ls="--", lw=1.2, label=f"mean = {mean:.0f} samples")
    ax1.axhline(median, color="tab:green", ls="-.", lw=1.1, label=f"median = {median:.0f} samples")
    n_floor = sum(1 for v in recovered if v == 0)
    ax1.annotate(f"{n_floor}/{len(recovered)} seeds: recovery = 0\n(never left the pre-drift band)",
                 xy=(0.03, 0.62), xycoords="axes fraction", fontsize=8.5,
                 color="tab:blue", va="top")
    ax1.axhline(RECOVERY_LOOKAHEAD, color="crimson", ls=":", lw=1.1)
    ax1.annotate(f"pre-registered budget = {RECOVERY_LOOKAHEAD:,}",
                 xy=(len(per_seed) - 0.6, RECOVERY_LOOKAHEAD), xytext=(0, 3),
                 textcoords="offset points", color="crimson", fontsize=8, ha="right")
    ax1.set_ylabel("Recovery time (samples from $\\tau$)")
    ax1.set_title("Streaming recovery time per seed")
    ax1.legend(fontsize=8, loc="upper left")
    ax1.grid(alpha=0.25, lw=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_xlabel("RNG seed")
    ax1.set_ylim(0, RECOVERY_LOOKAHEAD * 1.14)

    dr_pos = np.array([a if a and a > 0 else floor for a in dr_area], dtype=float)
    fr_pos = np.array([a if a and a > 0 else floor for a in fr_area], dtype=float)
    dr_zero = [i for i, a in enumerate(dr_area) if not (a and a > 0)]
    w = 0.4
    ax2.bar(x - w / 2, dr_pos, width=w, color="tab:blue", label="streaming")
    ax2.bar(x + w / 2, fr_pos, width=w, color="tab:red", label="frozen")
    zero_stubs(ax2, [x[i] - w / 2 for i in dr_zero], w)
    ax2.set_yscale("log")
    ax2.set_ylim(floor, max(fr_pos.max(), dr_pos.max()) * 2.2)
    ax2.set_ylabel("Damage area (samples·$\\kappa$, log scale)")
    ax2.set_title("Skill lost to the drift per seed")
    handles, _ = ax2.get_legend_handles_labels()
    if dr_zero:
        handles.append(Patch(facecolor="tab:blue", hatch="///", edgecolor="white",
                             label=f"streaming: damage 0 ({len(dr_zero)} seeds)"))
    ax2.legend(handles=handles, fontsize=8, loc="upper center", ncol=3, framealpha=0.95)
    ax2.grid(alpha=0.25, lw=0.5, axis="y")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=8)
    ax2.set_xlabel("RNG seed")
    fig.suptitle("Exp-02 — seed sweep: is the 835-sample recovery reproducible?", fontsize=12)
    save(fig, "exp02-seed-sweep-recovery.png")

    # --- 2. recovery time per seed, standalone ---------------------------------
    fig, ax = new_ax()
    ax.scatter(x, plotted, s=60, color="tab:blue", zorder=4, label="streaming recovery")
    ax.axhline(mean, color="tab:blue", ls="--", lw=1.2,
               label=f"mean = {mean:.0f} ± {sd(recovered):.0f} samples")
    ax.axhline(median, color="tab:green", ls="-.", lw=1.1, label=f"median = {median:.0f} samples")
    ax.axhline(RECOVERY_LOOKAHEAD, color="crimson", ls=":", lw=1.1)
    ax.annotate(f"pre-registered budget = {RECOVERY_LOOKAHEAD:,}", xy=(0.02, RECOVERY_LOOKAHEAD),
                xytext=(0, 3), textcoords="offset points", color="crimson", fontsize=8.5, ha="left")
    ax.annotate(f"{n_floor}/{len(recovered)} seeds sit on the metric floor\n"
                f"recovery = 0 means the model never left its pre-drift band",
                xy=(0.02, 0.60), xycoords="axes fraction", fontsize=8.5, color="tab:blue", va="top")
    seed_axis(ax)
    ax.set_ylabel("Recovery time (samples from $\\tau$)")
    ax.set_title("Exp-02 — streaming recovery time per seed\n"
                 "the metric floors at 0, so its mean is not the story")
    ax.set_ylim(0, RECOVERY_LOOKAHEAD * 1.14)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "exp02-recovery-per-seed.png")

    # --- 3. damage per seed, standalone ---------------------------------------
    fig, ax = new_ax(10.5, 5.4)
    ax.bar(x - w / 2, dr_pos, width=w, color="tab:blue", label="streaming")
    ax.bar(x + w / 2, fr_pos, width=w, color="tab:red", label="frozen")
    zero_stubs(ax, [x[i] - w / 2 for i in dr_zero], w)
    ax.set_yscale("log")
    ax.set_ylim(floor, max(fr_pos.max(), dr_pos.max()) * 2.6)
    handles, _ = ax.get_legend_handles_labels()
    if dr_zero:
        handles.append(Patch(facecolor="tab:blue", hatch="///", edgecolor="white",
                             label=f"streaming: damage exactly 0 ({len(dr_zero)} seeds)"))
    seed_axis(ax)
    ax.set_ylabel("Damage area (samples·$\\kappa$, log scale)")
    ax.set_title("Exp-02 — damage area per seed\n"
                 f"mean {statistics.mean(dr_area):.0f} ± {sd(dr_area):.0f} vs "
                 f"{statistics.mean(fr_area):.0f} ± {sd(fr_area):.0f} samples·κ")
    ax.legend(handles=handles, fontsize=8, loc="upper center", ncol=3, framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    save(fig, "exp02-damage-per-seed.png")

    # --- 4. the damage gap as a distribution ----------------------------------
    fig, ax = new_ax(8.5, 5.2)
    rng = np.random.default_rng(0)
    for pos, (vals, color, name) in enumerate(
            [(dr_area, "tab:blue", "streaming"), (fr_area, "tab:red", "frozen")]):
        at_floor = [v if v and v > 0 else floor for v in vals]
        jitter = rng.uniform(-0.08, 0.08, len(vals))
        ax.scatter(pos + jitter, at_floor, s=60, color=color, alpha=0.85, zorder=3,
                   label=f"{name} (one dot per seed)")
        med = statistics.median(vals)
        ax.hlines(max(med, floor), pos - 0.22, pos + 0.22, color=color, lw=2.5, zorder=4)
        ax.annotate(f"median {med:.3g}", xy=(pos, max(med, floor)), xytext=(0, 9),
                    textcoords="offset points", ha="center", fontsize=8.5, color=color)
    ax.annotate(f"{len(dr_zero)} streaming seeds measured exactly 0\n(plotted at the axis floor)",
                xy=(0.32, 0.05), xycoords="axes fraction", fontsize=8.5, color="tab:blue")
    ax.set_yscale("log")
    ax.set_ylim(floor, max(fr_area) * 3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["streaming", "frozen"])
    ax.set_ylabel("Damage area (samples·$\\kappa$, log scale)")
    ax.set_title("Exp-02 — the damage gap as a distribution\n"
                 f"streaming's worst seed still beats frozen's best by "
                 f"{ratio(min(fr_area), max(dr_area), '.0f')}×")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    save(fig, "exp02-damage-distribution.png")

    # --- 5. dip depth per seed, paired ----------------------------------------
    fig, ax = new_ax(9.5, 5.8)
    y = np.arange(n)
    for i in range(n):
        ax.plot([dr_dip[i], fr_dip[i]], [y[i], y[i]], color="gray", lw=1.0, alpha=0.7, zorder=2)
    ax.scatter(dr_dip, y, s=60, color="tab:blue", zorder=3, label="streaming")
    ax.scatter(fr_dip, y, s=60, color="tab:red", zorder=3, label="frozen")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_ylabel("RNG seed")
    ax.set_xlabel("Dip depth $\\Delta\\kappa$ (skill the drift knocked off)")
    ax.set_title("Exp-02 — how hard the drift hit each model, per seed\n"
                 f"frozen dips deeper on all {n} seeds "
                 f"({ratio(statistics.mean(fr_dip), statistics.mean(dr_dip))}× on the mean)")
    ax.grid(alpha=0.25, lw=0.5, axis="x")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, "exp02-dip-depth-per-seed.png")

    # --- 6. each model's fall from its own baseline ---------------------------
    fig, ax = new_ax(9.5, 5.8)
    off = 0.16
    for i in range(n):
        ax.plot([i - off, i - off], [dr_mink[i], dr_k0[i]], color="tab:blue", lw=1.3, alpha=0.65, zorder=2)
        ax.plot([i + off, i + off], [fr_mink[i], fr_k0[i]], color="tab:red", lw=1.5, alpha=0.65, zorder=2)
    ax.scatter(x - off, dr_k0, s=42, facecolor="white", edgecolor="tab:blue", zorder=3,
               label="streaming — before τ")
    ax.scatter(x - off, dr_mink, s=42, color="tab:blue", zorder=3, label="streaming — worst after τ")
    ax.scatter(x + off, fr_k0, s=42, facecolor="white", edgecolor="tab:red", zorder=3,
               label="frozen — before τ")
    ax.scatter(x + off, fr_mink, s=42, color="tab:red", zorder=3, label="frozen — worst after τ")
    ax.annotate(f"mean fall {statistics.mean([a - b for a, b in zip(dr_k0, dr_mink)]):.3f}",
                xy=(0.02, 0.97), xycoords="axes fraction", fontsize=8.5, color="tab:blue", va="top")
    ax.annotate(f"mean fall {statistics.mean([a - b for a, b in zip(fr_k0, fr_mink)]):.3f}",
                xy=(0.02, 0.90), xycoords="axes fraction", fontsize=8.5, color="tab:red", va="top")
    seed_axis(ax)
    ax.set_ylabel("Rolling Cohen Kappa ($w = 1{,}000$)")
    ax.set_title("Exp-02 — each model's fall from its own pre-drift baseline\n"
                 "frozen never climbs back; streaming barely leaves")
    ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(1.02, 1))
    save(fig, "exp02-kappa-collapse.png")

    # --- 7 & 8. full-run metrics per seed -------------------------------------
    for vals_dr, vals_fr, name, ylabel, fname in [
        (dr_kap, fr_kap, "full-run Cohen Kappa", "Full-run Cohen Kappa (48k samples)",
         "exp02-full-run-kappa.png"),
        (dr_acc, fr_acc, "full-run accuracy", "Full-run accuracy (48k samples)",
         "exp02-full-run-accuracy.png"),
    ]:
        fig, ax = new_ax(10.5, 5.2)
        ax.plot(x, vals_dr, "-o", color="tab:blue", ms=6, lw=1.6, label="streaming")
        ax.plot(x, vals_fr, "-o", color="tab:red", ms=6, lw=1.6, label="frozen")
        ax.fill_between(x, vals_fr, vals_dr, color="tab:blue", alpha=0.10)
        wins = sum(1 for a, b in zip(vals_dr, vals_fr) if a > b)
        ax.annotate(f"streaming ahead on {wins}/{n} seeds",
                    xy=(0.02, 0.06), xycoords="axes fraction", fontsize=8.5, color="tab:blue")
        seed_axis(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Exp-02 — {name} per seed")
        ax.legend(fontsize=8, loc="center", ncol=2, framealpha=0.95)
        save(fig, fname)

    # --- 9. why the recovery metric under-reports ------------------------------
    fig, ax = new_ax(9.5, 5.6)
    credited = late = clean = 0
    for i in range(n):
        breached = dr_dip[i] > 0.05 * dr_k0[i]
        has_recovery = dr_rec[i] is not None and dr_rec[i] > 0
        if has_recovery:
            ax.scatter(dr_dip[i], dr_rec[i], s=75, color="tab:green", zorder=4)
            credited += 1
            yy = dr_rec[i]
        elif breached:
            ax.scatter(dr_dip[i], 0, s=75, marker="X", color="tab:orange", zorder=4)
            late += 1
            yy = 0
        else:
            ax.scatter(dr_dip[i], 0, s=60, marker="o", facecolor="white",
                       edgecolor="tab:blue", zorder=4)
            clean += 1
            yy = 0
        ax.annotate(str(per_seed[i]["seed"]), xy=(dr_dip[i], yy), xytext=(5, 5),
                    textcoords="offset points", fontsize=8, color="dimgray")
    handles = [
        Line2D([], [], marker="o", ls="", ms=8, color="tab:green",
               label=f"breached early → recovery credited ({credited})"),
        Line2D([], [], marker="X", ls="", ms=8, color="tab:orange",
               label=f"breached after τ+500 → scored 0 ({late})"),
        Line2D([], [], marker="o", ls="", ms=8, mfc="white", mec="tab:blue",
               label=f"never left the band → 0 by definition ({clean})"),
    ]
    ax.annotate("the rule looks for the first sustained in-band run *from τ*,\n"
                "so a later excursion is recorded as no excursion at all",
                xy=(0.02, 0.60), xycoords="axes fraction", fontsize=8.5, color="dimgray", va="top")
    ax.set_xlabel("Dip depth $\\Delta\\kappa$ (did the model leave its band at all?)")
    ax.set_ylabel("Recovery time recorded (samples from $\\tau$)")
    ax.set_title("Exp-02 — why the recovery metric under-reports\n"
                 f"{late} of {n} seeds breached the band yet were scored 0")
    ax.set_ylim(-120, (max(recovered) if recovered else RECOVERY_LOOKAHEAD) * 1.35)
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25, lw=0.5)
    save(fig, "exp02-recovery-vs-dip.png")

    return written


def main() -> int:
    ap = argparse.ArgumentParser(description="Exp-02 seed sweep (SEA, Exp-01 replica per seed)")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help=f"comma-separated RNG seeds (default: {DEFAULT_SEEDS})")
    ap.add_argument("--limit", type=int, default=None,
                    help="stream length override for a dev smoke test (default: full 60k)")
    ap.add_argument("--device", default=None,
                    help="torch device for the DeepRiver net (default: cuda if available else cpu)")
    ap.add_argument("--require-cuda", action="store_true",
                    help="abort instead of falling back to CPU (the sweep wants the GPU)")
    ap.add_argument("--out", default=None, help="results dir override (default: experiments/exp02)")
    ap.add_argument("--figures-dir", default=None,
                    help="where the figure suite is written (default: "
                         "docs/Phase1:Experimentation/figures)")
    ap.add_argument("--no-figure", action="store_true", help="skip figure rendering")
    ap.add_argument("--figure-only", action="store_true",
                    help="re-render the figure suite from results.json (no GPU, no rerun)")
    args = ap.parse_args()

    try:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    except ValueError:
        print("--seeds must be a comma-separated list of integers", file=sys.stderr)
        return 2
    if not seeds:
        print("--seeds is empty", file=sys.stderr)
        return 2

    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out) if args.out else repo / "experiments" / "exp02"
    figures_dir = (Path(args.figures_dir) if args.figures_dir
                   else repo / "docs/Phase1:Experimentation/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"

    if args.figure_only:
        # Re-render from the committed artifact: no GPU, no 20-minute rerun. Figures
        # are regenerated from results.json, never hand-drawn.
        if not results_path.exists():
            print(f"--figure-only: no artifact at {results_path}", file=sys.stderr)
            return 2
        make_figures(json.loads(results_path.read_text())["per_seed"], figures_dir)
        return 0

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but torch.cuda.is_available() is False", file=sys.stderr)
        return 2
    if device.startswith("cuda"):
        print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)
    else:
        if args.require_cuda:
            print("--require-cuda: no CUDA device available", file=sys.stderr)
            return 2
        print("WARNING: running on CPU — the DeepRiver loop was ~11x slower than the "
              "RTX 3050 in Exp-01 (~1,089 s vs ~99 s per seed). Use --require-cuda to "
              "fail fast instead of running the whole sweep on CPU.", flush=True)

    if args.limit:
        # Dev slice: keep the full run's geometry — tau at 50% of the stream, warm-up
        # at 20% — so the drift still has a post-drift region to measure. Pinning tau
        # at the end of the slice leaves nothing after the drift and every metric
        # comes back n/a.
        n_total = args.limit
        warm_up = max(1, round(0.20 * n_total))
        tau = max(warm_up + 2, min(TAU, round(0.50 * n_total)))
        print(f"dev slice: n_total={n_total:,} warm_up={warm_up:,} tau={tau:,} "
              f"(full-run geometry scaled to the slice)", flush=True)
    else:
        n_total, warm_up, tau = N_TOTAL, WARM_UP, TAU

    scenario = {
        "stream": "SEA concepts (sum>8 -> sum>9), one continuous U(0,10)^3 draw",
        "n_total": n_total,
        "warm_up": warm_up,
        "tau": tau,
        "noise": NOISE,
        "seeds": seeds,
        "rolling_kappa_w": ROLL_W,
        "recovery_band": RECOVERY_BAND,
        "recovery_lookahead": RECOVERY_LOOKAHEAD,
        "recovery_stretch": RECOVERY_STRETCH,
    }
    models = {
        "streaming": {
            "name": "DeepRiver RollingClassifier",
            "module": "RNN(3->16) -> Linear(16->2), logits",
            "window_size": WINDOW,
            "lr": LR,
            "optimizer": "adam",
            "loss": "cross_entropy",
            "steps_per_sample": 1,
        },
        "frozen": {
            "name": "sklearn RandomForestClassifier",
            "n_estimators": 100,
            "warm_up_samples": warm_up,
            "update": "never after warm-up",
        },
    }

    exp01_recovery = read_exp01_recovery(repo)
    print(f"exp02 seed sweep — {len(seeds)} seeds, {n_total:,} samples each "
          f"(device: {device})", flush=True)

    per_seed: list[dict] = []
    sweep_start = time.perf_counter()
    for i, seed in enumerate(seeds, 1):
        t0 = time.perf_counter()
        res = run_one_seed(seed, n_total=n_total, warm_up=warm_up, tau=tau, device=device)
        per_seed.append(res)
        dm = res["drift_metrics"]
        print(f"  [{i:>2}/{len(seeds)}] seed {seed:>3}: "
              f"streaming recovery {_fmt(dm['deepriver']['recovery_samples'], '.0f')} "
              f"(dip {_fmt(dm['deepriver']['dip_depth'])}, "
              f"damage {_fmt(dm['deepriver']['damage_area'], '.0f')}) | "
              f"frozen {'never' if dm['frozen']['recovery_samples'] is None else dm['frozen']['recovery_samples']} "
              f"(damage {_fmt(dm['frozen']['damage_area'], '.0f')})  "
              f"[{time.perf_counter() - t0:.1f}s]", flush=True)

        # Persist after every seed so a crash/interrupt keeps the seeds already run.
        partial = {
            "experiment": "Exp-02",
            "question": "Is Exp-01's 835-sample recovery a fluke of one seed?",
            "status": "in progress",
            "device": device,
            "scenario": scenario,
            "models": models,
            "seeds": seeds,
            "seeds_done": [r["seed"] for r in per_seed],
            "acceptance": acceptance(aggregate(per_seed)),
            "aggregate": aggregate(per_seed),
            "per_seed": per_seed,
        }
        results_path.write_text(json.dumps(partial, indent=2))

    sweep_s = time.perf_counter() - sweep_start
    agg = aggregate(per_seed)
    artifact = {
        "experiment": "Exp-02",
        "question": "Is Exp-01's 835-sample recovery a fluke of one seed?",
        "status": "concluded" if len(seeds) >= 10 else "pilot (fewer than 10 seeds)",
        "spec": "docs/Reports/Results.md (Exp-02 row) · docs/Reports/Sept9.md (next milestone 1)",
        "protocol": "docs/Phase1:Experimentation/streaming-vs-batch-metrics.md §5 (seeds & statistics)",
        "baseline": "docs/Phase1:Experimentation/exp01-deepriver-vs-frozen-sea.md (seed 42)",
        "device": device,
        "scenario": scenario,
        "models": models,
        "seeds": seeds,
        "sweep_runtime_s": sweep_s,
        "process_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "acceptance": acceptance(agg),
        "cross_check_exp01": (cross_check(per_seed, exp01_recovery) if not args.limit
                              else {"skipped": "dev slice — geometry differs from Exp-01"}),
        "aggregate": agg,
        "per_seed": per_seed,
    }
    results_path.write_text(json.dumps(artifact, indent=2))
    print(f"\nsaved json: {results_path}")

    # --- the acceptance-bar readout --------------------------------------------
    dr = agg["streaming_recovery_samples"]
    fr = agg["frozen_recovery_samples"]
    print("\n=== Exp-02 summary — recovery time as a distribution ===")
    print(f"seeds: {agg['n_seeds']} | streaming recovered: {dr['n']}/{agg['n_seeds']}"
          f" | recovery: {_fmt(dr['mean'], '.0f')} ± {_fmt(dr['sd'], '.0f')} samples"
          f" (min {_fmt(dr['min'], '.0f')}, median {_fmt(dr['median'], '.0f')}, "
          f"max {_fmt(dr['max'], '.0f')})")
    print(f"frozen recovered: {fr['n']}/{agg['n_seeds']} (never: {fr['never']})")
    print(f"damage area: streaming {_fmt(agg['streaming_damage_area']['mean'], '.0f')} ± "
          f"{_fmt(agg['streaming_damage_area']['sd'], '.0f')} vs frozen "
          f"{_fmt(agg['frozen_damage_area']['mean'], '.0f')} ± "
          f"{_fmt(agg['frozen_damage_area']['sd'], '.0f')} samples·κ")
    print(f"full-run kappa: streaming {_fmt(agg['full_run']['deepriver_cohen_kappa']['mean'])} vs "
          f"frozen {_fmt(agg['full_run']['frozen_cohen_kappa']['mean'])}")
    acc = artifact["acceptance"]
    print(f"acceptance: >=10 seeds {acc['min_seeds_10']['pass']} | "
          f"streaming recovers every seed {acc['streaming_recovers_on_every_seed']['pass']} | "
          f"frozen never on every seed {acc['frozen_never_recovers_on_every_seed']['pass']}")
    cc = artifact["cross_check_exp01"]
    if cc.get("exp02_recovery_samples") is not None and cc.get("exp01_recovery_samples") is not None:
        print(f"seed 42 cross-check vs Exp-01: {cc['exp02_recovery_samples']} vs "
              f"{cc['exp01_recovery_samples']} samples (delta {cc['delta_samples']})")

    if not args.no_figure:
        make_figures(per_seed, figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
