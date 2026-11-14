from __future__ import annotations

from experiments.driftlab.config import ExperimentConfig
from experiments.driftlab.harness import _acceptance_verdicts


def make_config(rules: dict) -> ExperimentConfig:
    return ExperimentConfig(
        exp_id="t",
        title="",
        description="",
        streams=[],
        models=[],
        detectors=[],
        detector_hosts=[],
        detector_channels=[],
        metrics={},
        seeds=[0, 1],
        acceptance=rules,
    )


def model_summary(aue: dict[str, float]) -> dict:
    return {
        name: {
            "aue": {"mean": value},
            "damage_ref": {"mean": value},
        }
        for name, value in aue.items()
    }


def test_empty_acceptance_is_not_a_pass():
    verdicts = _acceptance_verdicts(make_config({}), {}, {}, [])
    assert verdicts["pass"] is None
    assert "nothing was asserted" in verdicts["note"]


def test_aue_ratio_is_evaluated():
    rules = {
        "baseline_model": "frozen",
        "treatment_model": "streaming",
        "min_aue_ratio": 2.0,
    }
    verdicts = _acceptance_verdicts(
        make_config(rules),
        model_summary({"frozen": 1.0, "streaming": 0.25}),
        {},
        [],
    )
    assert verdicts["aue_ratio"]["pass"] is True
    assert verdicts["pass"] is True
    assert "note" not in verdicts


def test_aue_ratio_fails_below_threshold():
    rules = {
        "baseline_model": "frozen",
        "treatment_model": "streaming",
        "min_aue_ratio": 5.0,
    }
    verdicts = _acceptance_verdicts(
        make_config(rules),
        model_summary({"frozen": 1.0, "streaming": 0.5}),
        {},
        [],
    )
    assert verdicts["aue_ratio"]["pass"] is False
    assert verdicts["pass"] is False


def test_missing_treatment_model_asserts_nothing():
    rules = {
        "baseline_model": "frozen",
        "treatment_model": "absent",
        "min_aue_ratio": 2.0,
    }
    verdicts = _acceptance_verdicts(make_config(rules), model_summary({"frozen": 1.0}), {}, [])
    assert verdicts["pass"] is None
    assert "note" in verdicts


def test_far_rule_uses_worst_channel():
    rules = {"max_far_per_1k": {"adwin": 0.1}}
    detectors = {
        "ht|adwin|raw": {"far_per_1k": {"mean": 0.05}},
        "hat|adwin|ewma": {"far_per_1k": {"mean": 0.5}},
    }
    verdicts = _acceptance_verdicts(make_config(rules), {}, detectors, [])
    assert verdicts["far_adwin"]["worst_mean_far_per_1k"] == 0.5
    assert verdicts["far_adwin"]["pass"] is False
    assert verdicts["pass"] is False
