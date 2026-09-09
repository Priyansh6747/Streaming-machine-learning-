"""Exp-01: DeepRiver rolling deep classifier vs frozen batch RandomForest on SEA.
Spec: docs/Phase1:Experimentation/exp01-deepriver-vs-frozen-sea.md
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from river.metrics import Accuracy, CohenKappa
from river.utils import Rolling
from sklearn.ensemble import RandomForestClassifier

# --- Scenario constants (pre-registered in the exp01 doc) ---------------------
N_TOTAL = 60_000        # stream length (samples)
WARM_UP = 12_000        # frozen model's training window (20%)
TAU = 30_000            # drift onset (samples)
NOISE = 0.1             # 10% label noise
SEED = 42               # single seed for Exp-01 (>=10-seed sweep is a later experiment)
WINDOW = 500            # DeepRiver rolling window (pre-registered default)
LR = 1e-3               # DeepRiver learning rate (pre-registered default)
ROLL_W = 1_000          # rolling-Kappa window
RECOVERY_BAND = 0.95    # recovery = rolling Kappa >= band * kappa0
RECOVERY_LOOKAHEAD = 5_000  # samples after tau in which recovery must happen
RECOVERY_STRETCH = 500       # consecutive samples at/above the band to count


class _RNNHead(torch.nn.Module):
    """Small RNN over the rolling window, Linear head to the two classes.

    Input: (window_size, batch=1, n_features) — the last window of samples.
    Output: logits (batch, 2). DeepRiver's RollingClassifier requires the first
    trainable layer to take n_features units (its feature-incremental padding
    logic depends on it), hence an RNN head rather than a flattened MLP.
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


def build_stream(n_total: int = N_TOTAL) -> list[tuple[dict[int, float], int]]:
    """One continuous SEA feature stream; the label rule switches abruptly at tau.

    river 0.25's SEA reseeds its internal RNG per generator, so concatenating two
    variants with the same seed replays identical feature vectors under the new
    label rule — a train-set memorisation confound. Instead: one uninterrupted
    draw of x ~ U(0,10)^3, labelled by concept 0 (sum > 8) before tau and
    concept 1 (sum > 9) from tau on.
    """
    rng = np.random.default_rng(SEED)
    n_pre = min(TAU, n_total)
    out: list[tuple[dict[int, float], int]] = []
    for t in range(n_total):
        x = {i: float(rng.uniform(0.0, 10.0)) for i in range(3)}
        label = x[0] + x[1] > (8.0 if t < n_pre else 9.0)
        if rng.random() < NOISE:
            label = not label
        out.append((x, int(label)))
    return out


def make_deepriver(device: str = "cpu") -> "RollingClassifier":
    from deep_river.classification import RollingClassifier

    class _TrainableRollingClassifier(RollingClassifier):
        """deep-river 0.3.4 runs the forward before `module.train()` in `_learn`,
        so after a predict (eval mode) cuDNN RNN backward errors out. Force train
        mode before the update step."""

        def learn_one(self, x, y):  # noqa: D102
            self.module.train()
            return super().learn_one(x, y)

    return _TrainableRollingClassifier(
        module=_RNNHead(n_features=3, n_hidden=16),
        loss_fn="cross_entropy",
        optimizer_fn="adam",
        lr=LR,
        device=device,
        seed=SEED,
        window_size=WINDOW,
        is_class_incremental=False,
        is_feature_incremental=False,
    )


def run(n_total: int = N_TOTAL, warm_up: int = WARM_UP, device: str = "cpu") -> dict:
    tau = min(TAU, n_total)  # drift falls off the end of a short dev slice

    # --- materialise the stream once; both models see identical samples --------
    stream = build_stream(n_total)
    n = len(stream)
    eval_from = warm_up

    # --- frozen batch model: fit once on the warm-up window --------------------
    X_all = np.array([list(x.values()) for x, _ in stream], dtype=np.float64)
    y_all = np.array([y for _, y in stream], dtype=np.int64)
    t_fit = time.perf_counter()
    frozen = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    frozen.fit(X_all[:warm_up], y_all[:warm_up])
    t_fit = time.perf_counter() - t_fit
    # batch-predict the whole stream once (frozen model never learns after warm-up)
    pred_frozen = frozen.predict(X_all).astype(int)

    # --- streaming model --------------------------------------------------------
    dr = make_deepriver(device)

    # --- prequential loop over the post-warm-up region --------------------------
    acc_dr, acc_fr = Accuracy(), Accuracy()
    kappa_dr, kappa_fr = CohenKappa(), CohenKappa()
    roll_dr = Rolling(CohenKappa(), window_size=ROLL_W)
    roll_fr = Rolling(CohenKappa(), window_size=ROLL_W)

    t_start = time.perf_counter()
    roll_dr_hist: list[tuple[int, float | None]] = []
    roll_fr_hist: list[tuple[int, float | None]] = []
    for t, (x, y) in enumerate(stream):
        if t < eval_from:
            # warm-up: only DeepRiver learns (rolling window must be warm by tau)
            dr.learn_one(x, y)
            continue
        # test
        yp_dr = dr.predict_one(x)
        acc_dr.update(y, yp_dr)
        kappa_dr.update(y, yp_dr)
        roll_dr.update(y, yp_dr)
        roll_dr_hist.append((t, roll_dr.get()))
        yp_fr = pred_frozen[t]
        acc_fr.update(y, yp_fr)
        kappa_fr.update(y, yp_fr)
        roll_fr.update(y, yp_fr)
        roll_fr_hist.append((t, roll_fr.get()))
        # train
        dr.learn_one(x, y)
        if t % 5_000 == 0:
            print(f"  t={t:6d}  rollingKappa dr={roll_dr.get():.3f} fr={roll_fr.get():.3f}",
                  flush=True)
    t_run = time.perf_counter() - t_start

    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    n_eval = n - eval_from
    return {
        "scenario": {
            "stream": "SEA concepts (sum>8 -> sum>9), one continuous U(0,10)^3 draw",
            "n_total": n,
            "warm_up": warm_up,
            "tau": tau,
            "noise": NOISE,
            "seed": SEED,
            "rolling_kappa_w": ROLL_W,
        },
        "models": {
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
        },
        "full_run": {
            "n_evaluated": n_eval,
            "deepriver": {"accuracy": acc_dr.get(), "cohen_kappa": kappa_dr.get()},
            "frozen": {"accuracy": acc_fr.get(), "cohen_kappa": kappa_fr.get()},
            "runtime_s": {"deepriver_loop": t_run, "frozen_fit": t_fit},
            "samples_per_s": n_eval / t_run if t_run else None,
            "maxrss_mb": mem_kb / 1024,
        },
        "series": {
            "t": [t for t, _ in roll_dr_hist],
            "rolling_kappa_deepriver": [v for _, v in roll_dr_hist],
            "rolling_kappa_frozen": [v for _, v in roll_fr_hist],
        },
    }


def drift_metrics(series_t: list[int], k_dr: list[float | None], k_fr: list[float | None],
                  tau: int = TAU) -> dict:
    """kappa0, dip depth, recovery time and damage area per model (exp01 doc §4)."""

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


def make_figure(series_t, k_dr, k_fr, out_png: Path, verdict: str, tau: int = TAU) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2), dpi=150)
    ax.axvline(tau, color="crimson", ls="--", lw=1.2)
    ax.text(tau, 0.02, f"drift $\\tau={tau:,}$", color="crimson", ha="center", fontsize=9)
    ax.plot(series_t, k_dr, color="tab:blue", lw=1.4, label="DeepRiver rolling (window 500)")
    ax.plot(series_t, k_fr, color="tab:red", lw=1.2, ls=":", label="Frozen RF (fit once on 12k)")
    ax.axvspan(tau, tau + RECOVERY_LOOKAHEAD, color="gray", alpha=0.08)
    ax.set_xlabel("Stream time (samples)")
    ax.set_ylabel(f"Rolling Cohen Kappa ($w = {ROLL_W:,}$)")
    ax.set_title("Exp-01 — DeepRiver rolling deep classifier vs frozen batch on SEA\n" + verdict)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.25, lw=0.5)
    ax.set_ylim(-0.05, 1.0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    print(f"saved figure: {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="stream length override (dev)")
    ap.add_argument("--out", type=str, default=None, help="results dir override (dev)")
    ap.add_argument("--device", type=str, default=None,
                    help="torch device for the DeepRiver net (default: cuda if available else cpu)")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device != "cpu":
        print(f"device: {device} ({torch.cuda.get_device_name(0)})", flush=True)

    repo = Path(__file__).resolve().parents[2]
    out_dir = Path(args.out) if args.out else repo / "experiments" / "exp01"

    if args.limit is not None:
        print(f"dev slice: n_total={args.limit} (no drift metrics/figure)", flush=True)
        results = run(n_total=args.limit, warm_up=min(WARM_UP, max(1, args.limit // 4)),
                      device=device)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "results.json").write_text(json.dumps(results, indent=2))
        print(f"saved json: {out_dir / 'results.json'}")
        return

    print("running exp01 …", flush=True)
    results = run(device=device)
    results["drift_metrics"] = drift_metrics(
        results["series"]["t"],
        results["series"]["rolling_kappa_deepriver"],
        results["series"]["rolling_kappa_frozen"],
        tau=results["scenario"]["tau"],
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"saved json: {out_dir / 'results.json'}")

    s = results["series"]
    dr = results["drift_metrics"]["deepriver"]
    fr = results["drift_metrics"]["frozen"]
    dr_verdict = "PASS" if dr["recovery_time"] is not None else "fail/inconclusive"
    fr_verdict = ("never" if fr["recovery_time"] is None
                  else f"{fr['recovery_samples']} samples")
    verdict = (
        f"H1 DeepRiver recovery: {dr_verdict} "
        f"(dip {dr['dip_depth']:.3f}, recovers after {dr['recovery_samples']} samples); "
        f"frozen: {fr_verdict}"
    )
    make_figure(s["t"], s["rolling_kappa_deepriver"], s["rolling_kappa_frozen"],
                repo / "docs/Phase1:Experimentation/figures/"
                       "exp01-rolling-kappa-deepriver-vs-frozen.png",
                verdict, tau=results["scenario"]["tau"])


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sys.exit(main())
