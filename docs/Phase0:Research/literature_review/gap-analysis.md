# Gap Analysis — The Delay Tax

Written after logging the index and per-paper notes. Maps every entry to the four
lemmas, confirms/revises the "active but fragmented" claim, flags scoop risk, and
recommends the baseline to reproduce.

---

## 1. Is the field "active but fragmented, no accepted benchmark"? — CONFIRMED, with one update

The claim in `idea/scope.md` holds, but must be sharpened in light of 2026 work:

- **Active:** Gomes et al. 2022 (CSUR survey), Alencar et al. 2026 (DSP
  experimental survey), Mello et al. 2021/2022 (verification latency in active
  learning), Botacin & Gomes 2024 (COSE label-delay measurement), plus the
  adjacent hybrid line (STUDD 2023; Gamma-Weibull 2026).
- **Fragmented:** confirmed. Terminology still split (verification latency,
  delayed labelling, label latency, extreme verification latency, delayed
  partially labelled streams). No canonical formulation, benchmark, or protocol —
  the Gomes survey's unified setting is a proposal, not yet the standard.
- **UPDATE (important):** the *immediate-label* side of the field has now gained
  a proper benchmarking framework — Cerqueira et al., **KDD'26** ("A Framework
  for Evaluating and Benchmarking Concept Drift Detection Methods"). So "no
  accepted benchmark" is **no longer true for the immediate-label detector
  benchmark**; it is still true for the **delayed/partial-label** detector
  benchmark. Our gap statement must be reworded to: *"there is now a standard
  benchmark for immediate-label drift detection, but no equivalent under delayed,
  partial, or selective labels."*

**Revised one-line gap:** the field lacks (a) a detector-level benchmark under
label-observation regimes, (b) formal bounds connecting label latency/coverage
to detection quality, and (c) a standard confirmation protocol for early-warning
+ delayed-confirmation hybrids.

---

## 2. Lemma-by-lemma coverage in the literature

### Lemma 1 — Reaction floor (detection cannot precede label arrival)
- **Partial treatment exists, empirically:**
  - Gomes et al. 2022: AGRAWAL experiment — 10,000-instance delay shifts
    detections by ~10,000 instances; prequential accuracy dips below 50%.
  - Botacin & Gomes 2024: delay epochs map ~linearly to exposure increase; 81
    epochs delay → +50% exposure; delay beyond that fully nullifies drift
    mitigation.
  - Joulani et al. 2013 (regret side): delay hurts learning multiplicatively
    (adversarial) / additively (stochastic) — the learning analogue of the floor.
- **No formal stopping-time bound** (`T ≥ τ + L_τ` pointwise) exists in this
  literature. The classical quickest-detection-with-delayed-observations framing
  (Page 1954 / Lorden 1971 / Shiryaev) is the mathematical root, but no one has
  stated the reaction floor as a theorem for error-stream detectors under
  verification latency.
- **Verdict: the theorem is novel as stated** (trivial to prove, novel to
  publish); the empirical phenomenon is already known — Ch3 must present the
  lemma as formalizing known behavior, and cite Gomes 2022 + Botacin 2024 as
  empirical corroboration.

### Lemma 2 — Power loss (`1/√p` sensitivity degradation under partial labels)
- **Adjacent treatments:**
  - Gomes et al. 2022 survey: SLT discussion — effective sample size = labelled
    count (the same intuition, not a detector-level law).
  - Alencar et al. 2026 (DSP): the experimental survey *likely* measures
    detector degradation as label availability decreases — **must be read in
    full**; if it reports a `1/√p`-like curve for ADWIN/DDM, our contribution
    shrinks to formalizing the factor (still publishable, but the empirics are
    taken).
  - STUDD 2023: sensitivity analysis over l_access ∈ {1,10,25,50}% shows
    relative method ordering changes with label access — qualitative, not a law.
- **Verdict: the `1/√p` closed-form bound for windowed error-stream detectors
  appears genuinely novel.** High-priority action: read Alencar 2026 to confirm.

### Lemma 3 — False-alarm inflation under batch arrival (design effect `1+(b−1)ρ`)
- **No direct treatment found.** Nothing in the logged literature studies batch/
  bursty label arrival injecting autocorrelation into the observed error stream
  and inflating detector false alarms. The clustered-data variance factor is
  classical statistics, but its application to drift detectors under
  batch-reconciled labels is **open**.
- Botacin & Gomes 2024 model *queue size* → *delay*, not batch correlation →
  false alarms; closest conceptual neighbor but not the same mechanism.
- **Verdict: the most clearly novel lemma.** Least prior risk.

### Lemma 4 — Identifiability ceiling (pure posterior drift invisible to label-free)
- **Already established.** Žliobaitė 2010 (ICDMW) proves the detectability
  conditions; Gomes et al. 2022 reproduce the table; STUDD 2023 empirically
  demonstrates the limitation (cannot catch posterior drift, explicitly stated).
- **Verdict: NOT novel as a claim.** The novelty in Ch3 must be the *use* of it:
  the hybrid confirmation protocol that makes it actionable (label-free early
  warning + delayed supervised confirmation is the information-maximal
  architecture). Present Lemma 4 as a formal restatement of Žliobaitė, credited
  properly.

---

## 3. SCOOP WARNINGS — read these before finalizing novelty claims

1. **Alencar et al. 2026 (DSP 182:106325) — HIGH.** An experimental survey of
   *exactly* our Ch3 setting (delayed + partially labeled streams, detector
   taxonomy, label-scarcity experiments). We have only verified it exists; we
   have NOT read its results. If it already contains delay-vs-latency curves and
   label-coverage-vs-false-alarm curves for ADWIN/DDM/KSWIN, part of our Ch3
   "measure the degradation" chapter is scooped as *measurement* — our remaining
   novelty is then the formal lemmas (esp. 1–3) and the hybrid protocol. **Action:
   read it in full immediately.**
2. **Botacin & Gomes 2024 (COSE) — MEDIUM-HIGH.** Already quantified label-delay
   cost on real malware pipelines + pseudo-label mitigation. Do not claim: "we
   quantify that delayed labels hurt." Do claim: detector-level isolation, formal
   bounds, confirmation protocol, synthetic ground-truth delay curves. Also do
   not claim a novel time-aware metric without contrasting with exposure/AUT.
3. **Gomes et al. 2022 (CSUR) — MEDIUM.** The unified setting and the
   delayed-label AGRAWAL illustration cover our formal framing at a survey level.
   They explicitly flag "forget outdated labels after drift" and the
   under-explored intersection (iv) as open — cite them as naming our gap.
4. **Agu et al. 2026 (JKSU-CIS) — LOW-MEDIUM.** Pseudo-label adaptation gated by
   a statistical detector is architecturally close to our hybrid, but it
   sidesteps delayed ground truth rather than measuring it. Distinguish clearly.
5. **Mello et al. 2021/2022 — LOW.** Verification latency in *active learning*
   overlaps our selective-labeling regime; not detector isolation.

**Net position:** the *combination* of (formal lemmas 1–3) + (detector-level
delay/coverage/batch curves on synthetic ground truth) + (a confirmation
protocol, evaluated against naive-delayed, label-free-only, and immediate-label
controls) remains unclaimed. The biggest single risk is #1.

---

## 4. Which published baseline to reproduce for Phase 0

**Recommendation: Cerqueira et al., KDD'26 — "A Framework for Evaluating and
Benchmarking Concept Drift Detection Methods."**

Why:
- It is the most recent (2026), venue-strong (KDD, open access, CC-BY, code
  public), and directly relevant: reproducing one of its detector comparisons
  (e.g., ADWIN vs DDM vs KSWIN on one injected-drift real dataset) exercises
  exactly the harness + metrics Ch1 needs.
- It is *immediate-label*; that's fine for Phase 0 (we need the control numbers
  first) and it lets us later *extend* it with delay regimes in Ch3, which is a
  clean narrative: "we take the field's new standard benchmark and break it with
  delayed labels."
- Feasibility note: it uses MOA-family methods (Java) for some detectors; River
  and CapyMOA both implement ADWIN/DDM/EDDM/KSWIN/Page-Hinkley, so a River
  reproduction of a subset (2–3 detectors, 1–2 datasets, abrupt + gradual) is
  realistic.

Fallback alternatives (if KDD'26 proves too heavy to reproduce):
- **Gonçalves et al. 2014 (ESWA)** — the classic detector comparison; reproducing
  a headline number (e.g., no detector dominates on SEA) is cheap and citable.
- **Gomes et al. 2022 survey's AGRAWAL experiment** — reproduce the delayed-label
  shift (~10k instances) as a sanity check that our harness correctly models
  delay; this doubles as a first Ch3 pilot result.

**Recommended Phase 0 checklist:**
1. Pin River version; verify ADWIN/DDM/EDDM/KSWIN/PHT implementations exist.
2. Reproduce one KDD'26-style detector comparison on a real dataset with injected
   drift (or fallback: one ESWA-2014-style SEA comparison).
3. As a stretch: reproduce Gomes 2022's delayed-AGRAWAL shift as the first
   delayed-label pilot.

---

## 5. What stays in the project (post-analysis)

| Item | Status |
|---|---|
| Ch1 baseline benchmark (immediate labels) | Valid; now has a reference protocol (KDD'26) to build on |
| Ch2 systems/ops chapter (replica divergence, backpressure, restart cost) | Untouched by the whole Ch3 literature — still our most defensible novelty, zero scoop risk |
| Ch3 lemmas 1–3 (floor, 1/√p, batch inflation) | Novel as formal detector-level statements; empirics partially adjacent (Alencar 2026 to check) |
| Ch3 lemma 4 (identifiability) | Žliobaitė's result; ours only as restatement + hybrid motivation |
| Ch3 hybrid confirmation protocol | Architecture named as open by Gomes 2022 and STUDD 2023; no built/evaluated instance found — **the central claim** |
| Delayed-label evaluation protocol | Grzenda et al. 2020 (continuous re-evaluation) is the standard to use, not reinvent |

---

## 6. Ch1 status update — direct streaming-vs-batch comparisons (added Sep 2026)

After web research logged into `index.md` §1 (subgroup "Streaming vs batch /
retraining-policy comparisons"), the landscape for the Ch1/Phase-1 thesis question
("take non-streaming models and beat them on real metrics") is now clear:

**Direct comparisons already exist — Ch1 is not novel as a paradigm comparison.**

- Read et al. 2012 (IDA): instance-incremental vs batch-incremental under drift —
  no universal winner.
- Chen 2019 (SEAMS): retrained vs incremental on 1,360 conditions — winner depends
  on algorithm and drift intensity; incremental faster, often equal-or-better.
- Amekoe et al. 2024: tuned frequent-retrain batch beats streaming under delay +
  rare class (already logged).
- Applied 2024–2026 (Hans & Zühlke 2026 Water Supply; Abdoune et al. 2026 ESWA;
  Bartz-Beielstein & Hans 2024): online/incremental wins operationally under drift
  intensity and on runtime.

**Scope-defining new evidence — Dasari 2026 (arXiv:2608.19488):**

- With per-sample *incremental* updates, retraining policy is irrelevant (54
  paired comparisons, no practical difference vs no-retrain).
- Without incremental updates, **periodic retraining beats both reactive
  (drift-triggered / error-threshold) policies** under abrupt and gradual drift;
  reactive only wins under recurring drift.
- ⇒ The thesis claim must be worded about **streaming learners**, not about drift
  detectors used as batch-refit triggers. "Event-driven beats clock-driven" is only
  true when the *learner* updates per sample; a detector firing at batch cadence
  inherits the clock's weaknesses. Our thesis.md §1/§3 mechanism (bounded recovery
  via per-sample updates) already reflects this; keep it explicit.

**What remains ours for Ch1 (defensible, narrow):**

1. The **reactivity-metric family** applied to this comparison — recovery time,
   damage area, rolling-Kappa dip depth, detection delay — none of the found
   studies measure these (they use mean accuracy/MAE and runtime).
2. A **fair, pre-registered protocol** with frozen + scheduled + oracle-tuned
   baselines, ≥10 seeds, Friedman + Nemenyi, and an explicit win card (one claim
   per metric, dataset + number).
3. Quantifying the **scheduled-batch phase-noise** effect (recovery time scatters
   uniformly over the retrain interval) rather than reporting one lucky schedule.
4. The Delay Tax (Ch3) remains the project's main novelty; Ch1 is best framed as
   the rigorous baseline chapter the literature lacks (consistent with KDD'26 as
   the detector-side benchmark to build on).
