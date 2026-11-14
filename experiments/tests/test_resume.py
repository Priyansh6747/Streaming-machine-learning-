from __future__ import annotations

from pathlib import Path

from experiments.driftlab.config import ExperimentConfig, StreamConfig
from experiments.driftlab.harness import run_experiment


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="t01",
        title="resume test",
        description="",
        streams=[
            StreamConfig(
                generator="sea",
                shape="abrupt",
                n_samples=800,
                warm_up=300,
                noise=0.1,
                position=500,
                concept_a=0,
                concept_b=1,
            )
        ],
        models=[
            {
                "id": "ht",
                "type": "river",
                "params": {"class": "HoeffdingTreeClassifier"},
            }
        ],
        detectors=[],
        detector_hosts=[],
        detector_channels=[],
        metrics={
            "rolling_window": 100,
            "band": 0.95,
            "stretch": 50,
            "window": 200,
            "delta_max": {"abrupt": 200},
        },
        seeds=[0],
        acceptance={},
    )


def test_resume_skips_completed_units(tmp_path: Path):
    config = make_config()
    out = tmp_path / "t01"

    first = run_experiment(config, repo=tmp_path, out_dir=out, seeds=[0])
    assert first["n_model_runs"] == 2
    assert (out / "cache" / "s0_seed0.json").exists()

    second = run_experiment(config, repo=tmp_path, out_dir=out, seeds=[0], resume=True)
    assert second["n_model_runs"] == 2
    assert len((out / "models.jsonl").read_text().splitlines()) == 2
    assert len((out / "episodes.jsonl").read_text().splitlines()) == 2
    assert (out / "meta.json").read_text().count('"resumed": true') == 1


def test_resume_rejects_changed_config(tmp_path: Path):
    config = make_config()
    out = tmp_path / "t01"
    run_experiment(config, repo=tmp_path, out_dir=out, seeds=[0])

    changed = make_config()
    changed.title = "different"
    try:
        run_experiment(changed, repo=tmp_path, out_dir=out, seeds=[0], resume=True)
    except SystemExit as exc:
        assert "config hash changed" in str(exc)
    else:
        raise AssertionError("resume should refuse a changed config")
