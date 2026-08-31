# 07 — Open questions (decide these before planning)

Things that change what the project *is*. Not implementation details.

## 1. Which direction? ⭐ the big one

See [05 — Project directions](05-project-directions.md). Current recommendation:
**Option C** (detector benchmark + does-adaptation-pay-off), with **Option F**
(a hybrid meta-detector) as an optional stretch chapter.

Nothing else can be settled until this is.

- [ ] Decided: ______________________

## 2. What does your teacher actually want?

The word "research-based" is doing a lot of unspecified work. Worth asking directly,
because the answer changes the shape of everything:

- Is the deliverable a **paper/report**, a **working system**, or both?
- Expected length? Is there a required format (IEEE two-column is common)?
- Is a **literature review** a required chapter, or just background?
- Is **novelty** required, or is a rigorous reproduction/comparison acceptable?
  → This single answer decides whether Option F is mandatory or optional.
- Individual or group? Deadline? Any interim milestones?
- Is River **required**, or just a suggestion? (It's a good choice either way — but
  if MOA is acceptable too, that's worth knowing, since more algorithms live there.)

- [ ] Asked
- [ ] Answers recorded here: ______________________

## 3. Synthetic only, real only, or both?

**Recommendation: both, and say why.** Synthetic streams give ground-truth drift
positions (so detection delay is computable); real datasets give external validity.
Using only synthetic invites "but does this matter in practice?"; using only real
makes half the interesting metrics impossible.

- [ ] Decided: ______________________

## 4. Scope guard — what are you explicitly NOT doing?

Writing this down early prevents scope creep later. Candidates to rule out:

- Regression and clustering (stick to **classification**?)
- Deep learning / `deep-river`
- Distributed or production deployment (Kafka, Flink)
- Unsupervised drift detection (no labels available)
- The delayed/partial-labels problem — realistic, but a whole project on its own

- [ ] Out of scope: ______________________

## 5. Do you need a "system," or just experiments?

Some teachers want something that *runs* — a live dashboard, a demo where a stream
plays and you watch accuracy dip and recover as drift hits. That's a genuinely
compelling demo and not much extra work on top of Option C, since the data is
already being logged per-step.

- [ ] Needed? ______________________

## 6. Practical constraints

- Machine: RAM, CPU cores? (Option C is laptop-scale; wide sweeps × 10 seeds are
  where it adds up.)
- Time budget: weeks available?
- Comfortable with Python, pandas, matplotlib? Any of these needing learning time?
- Version control / backups sorted?

---

## Once 1–4 are answered

Then we plan: environment setup, experiment harness design, the run matrix, and the
report skeleton. **Not before** — planning against an undecided question wastes the
plan.
