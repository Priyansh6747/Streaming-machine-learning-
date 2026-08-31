# Deep research prompts (Perplexity / any deep-research tool)

## How to use these

Run them **in order**, one at a time, in separate threads. Save each output into
`research/findings/` as `01-landscape.md`, `02-delayed-labels.md`, etc.

Don't paste all five at once — deep research tools do noticeably better with one
focused brief than one sprawling one.

**The one thing you must do:** treat every citation as unverified until you've seen
it. These tools hallucinate plausible-looking papers — right-sounding title, real
author names, wrong or nonexistent paper. Check each one on Google Scholar or
Semantic Scholar before it enters your bibliography. A fake citation in a submitted
report is the kind of thing that ends badly.

**What you're actually looking for:** Prompt 2 is the important one. It's testing
whether the gap this project bets on is real. If it comes back saying the delayed-label
problem is thoroughly solved, we change direction — and finding that out in week 2
costs nothing, whereas finding out in month 8 costs everything.

---

## Prompt 1 — The landscape

```
You are conducting a literature survey for a one-year Master's-level research
project on online/streaming machine learning and concept drift adaptation.

Produce a comprehensive, structured survey of the field. Cover:

1. FOUNDATIONS
   - The origins of data stream mining and online learning
   - Key definitions: online vs incremental vs batch learning; concept drift vs
     virtual drift / covariate shift
   - The formal taxonomy of drift types (sudden, gradual, incremental, recurring)
     and which papers established it

2. ALGORITHMS
   - Foundational streaming classifiers (Hoeffding Trees / VFDT, CVFDT and
     successors) — original papers and the key improvements since
   - Ensemble methods for streams (Adaptive Random Forest, Streaming Random
     Patches, Leveraging Bagging, online bagging/boosting)
   - The current state of the art for streaming classification as of 2026, and how
     the field has moved since the 2017-2020 wave

3. DRIFT DETECTION
   - Major supervised detectors (DDM, EDDM, ADWIN, Page-Hinkley, HDDM, KSWIN,
     FHDDM) — mechanism, assumptions, known weaknesses
   - Unsupervised / label-free drift detection methods that monitor the input
     distribution rather than the error stream
   - Comparative studies that benchmark detectors against each other, and what
     they concluded

4. EVALUATION METHODOLOGY
   - Prequential / test-then-train evaluation and its theoretical justification
   - Known criticisms and limitations of standard streaming evaluation practice
   - Standard benchmark datasets (both real and synthetic generators) and any
     published critiques of them — for example, known problems with the
     Electricity/Elec2 dataset

5. TOOLING
   - River, MOA, scikit-multiflow, capymoa, and any newer frameworks
   - Their relative maturity, adoption, and which the research community
     actually benchmarks on

For each major work give: full citation, year, venue, the core contribution in
1-2 sentences, and its influence (roughly, citation count tier).

Prioritise peer-reviewed papers, surveys, and textbooks over blog posts.
Explicitly flag where the community disagrees or where results conflict.
Note which findings are well-replicated versus based on a single paper.

Format as structured markdown with clear sections and a full reference list.
```

---

## Prompt 2 — Delayed and partial labels ⭐ the critical one

```
I am researching a specific gap in the concept drift literature: the assumption
that ground-truth labels arrive immediately alongside each sample.

In real deployments labels are delayed (fraud confirmed weeks later via
chargeback), partial (only some samples ever get labelled), biased (only
flagged cases are reviewed), or never arrive at all.

Investigate thoroughly and tell me the current state of this problem:

1. HOW MUCH WORK EXISTS
   - What is published on learning from data streams with delayed, partial,
     scarce, or verification-latency-affected labels?
   - What terminology does the literature use? I have seen "verification
     latency", "delayed labelling", "label latency", "extreme verification
     latency" — identify all standard terms, since inconsistent naming may hide
     relevant work
   - Is there a survey specifically on this problem?

2. DETECTION WITHOUT TIMELY LABELS
   - How do drift detectors perform when the error stream they monitor is
     delayed? Has anyone quantified this degradation?
   - What unsupervised/label-free drift detection methods exist, and how do they
     compare to supervised detectors in detection delay and false alarm rate?
   - Has anyone proposed hybrid approaches that combine a label-free early
     warning with delayed supervised confirmation? If so, what did they find?

3. EVALUATION UNDER DELAY
   - Prequential evaluation assumes immediate scoring. How does the literature
     evaluate models when labels are delayed?
   - Are there accepted protocols or benchmarks for this setting, or is each
     paper improvising?

4. ADJACENT FIELDS
   - What do semi-supervised streaming, active learning on streams, and
     self-training/pseudo-labelling contribute here?
   - Is there transferable work from positive-unlabelled learning?

5. THE GAP — this is what I most need
   - What specifically remains unsolved?
   - Which open problems do recent papers name in their future-work sections?
   - Is this an active area with recent (2023-2026) activity, or largely dormant?

Be direct and evidence-based about how much prior work exists. If this problem
is already well solved, say so plainly and point me to the work that solves it —
I would much rather learn that now than after committing a year to it.

Give full citations with venues and years. Format as structured markdown.
```

---

## Prompt 3 — Streaming ML as a production system

```
I am researching the engineering and operational side of deploying continually
learning (online/streaming) machine learning models in production, as opposed
to the algorithmic side.

Investigate:

1. PUBLISHED ARCHITECTURES
   - Papers, industry engineering blogs, or case studies describing real
     production deployments of online/incrementally-learning models
   - Reference architectures for continual learning systems, and how model
     serving is combined with continuous model updating
   - Which companies have publicly described doing this, and what they learned

2. THE OPERATIONAL PROBLEMS
   A batch model is an immutable versioned artifact. A continually-learning
   model is mutating state. What does the literature say about:
   - Versioning and reproducibility of a model whose state depends on every
     event it has ever seen, in order
   - Rollback when there is no previous artifact to roll back to
   - Checkpointing and recovery of model state
   - Horizontal scaling: if you run N replicas that each learn independently,
     they diverge. How is this handled? Is there published work measuring
     divergence?
   - Auditability and regulatory compliance for a model whose past state no
     longer exists
   - A/B testing and shadow deployment when both variants keep changing

3. PERFORMANCE AND COST
   - Has anyone rigorously measured the latency, throughput, memory, or
     compute cost of online learning in a serving system (not offline
     simulation)?
   - What is the measured cost of drift adaptation — model resets, background
     retraining — on serving latency?
   - Comparisons of total cost of ownership: continual updating vs periodic
     batch retraining

4. MONITORING AND MLOps
   - How does ML monitoring tooling (Evidently, WhyLabs, Arize, NannyML, Fiddler)
     handle drift, and how does their approach differ from the academic drift
     detection literature? Do the two communities cite each other at all?
   - Is there a gap between what academic drift detection assumes and what
     production monitoring tools actually do?

5. THE GAP
   - Is the operational side of streaming ML under-researched relative to the
     algorithmic side?
   - What would a rigorous systems-level study of streaming ML need to measure
     that nobody has measured?

Include peer-reviewed work, industry engineering blogs, and conference talks
(MLSys, KDD applied track, QCon, Strange Loop). Clearly distinguish
peer-reviewed sources from industry blog posts.

Format as structured markdown with citations.
```

---

## Prompt 4 — Benchmarking methodology

```
I am designing rigorous benchmark experiments comparing concept drift detectors
and adaptive streaming classifiers. I want to follow the field's accepted
methodology rather than invent my own.

Research and report:

1. EXPERIMENTAL PROTOCOL
   - How do the best papers in this area design their experiments?
   - Standard practice for: number of random seeds, stream lengths, drift
     positions, how results are aggregated
   - How is a drift "successfully detected" formally defined? What tolerance
     window is standard, and how much does this vary between papers?

2. METRICS
   - Standard metrics for evaluating drift DETECTORS: detection delay, false
     positive rate, missed detection rate, MTFA, MTD, MDR — give precise
     definitions and the papers that established them
   - Are there composite metrics that combine detection quality into a single
     score?
   - Standard metrics for adaptive CLASSIFIERS under drift: recovery time,
     accuracy degradation, area under the accuracy-over-time curve

3. STATISTICAL TESTING
   - Accepted practice for comparing many methods across many datasets
   - Friedman test + Nemenyi post-hoc and critical difference diagrams — is this
     still the standard in 2026, or has practice moved on?
   - Common statistical mistakes in streaming ML papers

4. DATASETS AND GENERATORS
   - The standard synthetic generators (SEA, Agrawal, Hyperplane, RandomRBF,
     STAGGER, LED) — what drift type each is used for and their parameters
   - How researchers configure ConceptDriftStream-style setups to produce
     sudden vs gradual vs incremental vs recurring drift
   - Standard real-world benchmark datasets, and published critiques of them —
     especially any known flaws in Electricity/Elec2, Airlines, or KDDCup99

5. REPRODUCIBILITY
   - Known reproducibility problems in this literature
   - Have any papers attempted to replicate published streaming ML results, and
     what did they find?

Give concrete, actionable methodological guidance I can follow directly, with
citations for each convention.

Format as structured markdown.
```

---

## Prompt 5 — Gap hunting

*Run this last, after you've read the output of 1–4.*

```
Based on the current state of research in online/streaming machine learning and
concept drift adaptation, identify the most promising open research directions
for a one-year Master's-level project.

Context on me: my background is infrastructure, DevOps, distributed systems and
backend engineering, NOT machine learning theory. I am strong at building
systems, benchmarking, load testing, and measurement. I am weak at deriving
novel algorithms or proving theoretical bounds. A good project for me should
play to systems strengths rather than mathematical ones.

Please:

1. List open problems explicitly named in the future-work sections of recent
   (2023-2026) papers and surveys in this area

2. Identify where the field's standard assumptions diverge most sharply from
   real deployment conditions

3. Assess specifically: is there a meaningful research gap in the SYSTEMS and
   OPERATIONAL aspects of streaming ML — deployment architecture, latency and
   throughput cost of adaptation, state management, scaling, reproducibility —
   as opposed to the algorithmic aspects? Or is this already well covered?

4. For each direction you identify, assess:
   - How much prior work exists (is it crowded or open?)
   - Whether it is feasible in one year at part-time intensity
   - Whether it suits a systems engineer rather than an ML theorist
   - What the concrete deliverable and headline result would look like
   - The main risk of it failing

5. Be honest about which directions are already saturated and not worth entering

Rank the directions by suitability for my specific background and constraints.

Format as structured markdown with citations.
```

---

## After you run these

1. Save all five outputs to `research/findings/`
2. **Verify every citation you intend to use** — Google Scholar or Semantic Scholar
3. Pull the 10–15 most important papers and actually read them (abstract + intro +
   conclusion is enough for most; read 3 or 4 properly)
4. Write `research/gap-analysis.md`: does the delayed-labels + systems gap hold up?
5. **Then** we lock the scope and plan Phase 1

If Prompt 2 or Prompt 5 says the gap is already filled, we pivot. That's a normal
research outcome, not a setback — it's exactly what Phase 0 exists to find out.
