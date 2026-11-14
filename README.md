# Streaming Machine Learning — the Delay Tax

Models that never stop learning, on data that never stops changing — and the
price of learning from yesterday's truth.

A research project on **machine learning from data streams**: single-pass,
prequential learning in bounded memory, concept-drift detection and adaptation.
Built on [River](https://riverml.xyz/) with PyTorch learners via DeepRiver.

**Status:** Phase 0 (research) complete. Phase 1 (experimentation) has
**12 concluded experiments**; the surviving headline is the **Delay Tax** (E21).
E18/E19 failed their pre-registered gates. Full plan:
[Roadmap v2](docs/Reports/Roadmap-v2.md).

---

## What the evidence says

- **Streaming heals, frozen does not.** Across 11 seeds the frozen model recovered
  on **0/11**; corrected damage is **74.8 vs 515.0** samples·κ (≈6.9×, ≈10.8× over
  the rest of the stream). Recovery on 8/11 seeds ([E01](docs/Reports/E01-metric-v2.md)).
- **Scheduled retraining beats frozen, streaming beats both.** Leveraging Bagging
  455.7, HAT 489.2, ARF 575.4 vs best schedule 733.5 and frozen 871.7
  ([E03](docs/Reports/Results.md)).
- **Detector defaults are host-conditional.** HDDM-A on the raw signal wins the
  static synthetic host; on real Elec2 with an adaptive HAT, ADWIN wins abrupt and
  recurring. The KDD'26 reproduction ranks SEED/STEPD first ([E04](docs/Reports/E04-detector-map.md),
  [E17](docs/Reports/E17-real-detector-map.md)).
- **Deep signals are a warning channel, not a detector.** Best deep AUC **0.616**
  vs error-rate **0.598** (gate failed, [E18](docs/Reports/E18-deep-signals.md));
  DF-H v1 ranked 3rd, not top-2 ([E19](docs/Reports/E19-hybrid-detector.md)), but
  [E22](docs/Reports/E22-detector-shootout.md) shows it **3rd overall**, ahead of
  ADWIN, HDDM-A, HDDM-W and PageHinkley, at 0.04 FAR.
- **The label tax is real.** At `L = 2,000`, a label-free Stage 1 warns in **70–513**
  samples while error-only detectors take **2,120–3,269** ([E21](docs/Reports/E21-delay-detector.md)).

Earlier magnitude claims (**294×**, **24×**) were upper bounds from a truncated
metric and are retired. Every number traces to `experiments/results/<exp_id>/`.

---

## Layout

| Path | What's inside |
|---|---|
| `experiments/` | DriftLab harness, experiment configs, artifacts, tests ([README](experiments/README.md)) |
| `docs/Phase0:Research/` | Background notes, literature review, synopsis |
| `docs/Phase1:Experimentation/` | Thesis argument, protocols, experiment write-ups |
| `docs/Reports/` | [Results](docs/Reports/Results.md), milestone and per-experiment reports |

## Quickstart

```bash
uv sync
uv run pytest experiments/tests -q

# smoke test (writes to a scratch artifact)
uv run python -m experiments.run --exp e02 --seeds 42 --limit 1500 --no-proba \
    --out experiments/results/_smoke/e02

# full experiment, e.g. the corrected metric
uv run python -m experiments.run --exp e01

# docs site
uv run mkdocs serve
```

## Read more

- **[Results](docs/Reports/Results.md)** — consolidated verdicts for every experiment
- **[Thesis](docs/Phase1:Experimentation/thesis.md)** — why streaming wins structurally
- **[Roadmap v2](docs/Reports/Roadmap-v2.md)** — what's next
- **[Synopsis](docs/Phase0:Research/idea/synopsis.md)** — the original idea and scope
