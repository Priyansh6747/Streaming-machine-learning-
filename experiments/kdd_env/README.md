# Isolated environment for the KDD'26 benchmark reproduction.

CapyMOA requires `tqdm ~=4.67`, while the main project's `deep-river==0.3.4`
pins `tqdm <4.67`. Keeping CapyMOA out of the main environment avoids making
the main `uv sync` unsatisfiable.

```bash
uv sync --project experiments/kdd_env
./experiments/kdd.sh check
./experiments/kdd.sh run --datasets Electricity,Covtype --trials 5 --workers 8
```

The runner lives in `experiments/kdd_bench.py`.
